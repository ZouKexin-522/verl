import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
import requests
import time
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import http.client

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verl.utils.http_utils import http_post


def detect_concatenated_repeats(text: str, min_repeats: int = 10, min_length: int = 10) -> list:
    """Detect concatenated repeating segments in text (deadlock detection).

    Splits text by newlines and checks each line for repeating patterns
    to avoid regex memory pressure on large texts.
    """
    texts = text.split("\n")
    pattern = re.compile(r"(.{2,})\1{" + str(min_repeats - 1) + ",}")
    matches = []
    for t in texts:
        if len(t) <= min_length:
            continue
        match = pattern.search(t)
        if match:
            loop_kernel = match.group(1)
            if not any(c.isdigit() for c in loop_kernel):
                loop_kernel_char_set = set(loop_kernel)
                if " " in loop_kernel_char_set:
                    loop_kernel_char_set.remove(" ")
                if len(loop_kernel_char_set) > 1:
                    matches.append({"pattern": loop_kernel, "full_match": match.group(0), "span": match.span()})
                    break
    return matches


@dataclass
class RequestData:
    """Data class for request data returned by create_request."""

    success: bool
    reward_request: dict
    result_request: dict
    run_id: str


class ErnieXVerifier():
    """Handler for calculating rewards in pipeline"""

    """Async reward calculation core logic"""

    def __init__(self, config, *args, **kwargs):

        # print(f"Initializing ErnieXVerifierHandler, received config: {config}")

        # parse url
        try:
            self.reward_urls = config["reward_urls"]
            # print(f"Successfully obtained reward_urls: {self.reward_urls}")
        except KeyError:
            # print("Required reward_urls field missing in configuration")
            raise

        self.task_url = self.reward_urls[0]
        self.submit_url = self.reward_urls[1]
        self.result_url = self.reward_urls[2]

        self.default_error_reward = config.get("default_error_reward", -1.0)
        self.enable_thinking = config.get("enable_thinking", True)
        self.reward_protocol = config.get("reward_protocol", "normal")
        self.reward_auth_key = config.get("reward_auth_key")
        self.need_deadlock_check = config.get("need_deadlock_check", 0)
        self.chat_template_format = config.get("chat_template_format", "ERNIE")

        if self.reward_auth_key is not None:
            self.request_headers = {"X-API-Key": self.reward_auth_key, "Content-Type": "application/json"}
        else:
            self.request_headers = None

        self.task_id = self._get_task_id()  # must after self.request_headers init
        # 添加 _custom_fields 用于存储 max_tokens 等配置
        self._custom_fields = config.get("_custom_fields", {})
        # print(f"ErnieXVerifierHandler initialization completed, task_id: {self.task_id}")

    def _get_task_id(self):
        """Initialize task ID with retry mechanism."""
        # Configure session with retry strategy
        retry_strategy = Retry(
            total=5,  # total number of retries
            backoff_factor=1,  # wait 1, 2, 4, 8, 16 seconds between retries
            status_forcelist=[429, 500, 502, 503, 504],  # retry on these HTTP status codes
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Retry with exponential backoff on connection errors
        for attempt in range(5):
            try:
                if self.request_headers:
                    resp = session.get(self.task_url, headers=self.request_headers, timeout=30)
                else:
                    resp = session.get(self.task_url, timeout=30)
                resp.raise_for_status()
                result = resp.json()
                if "data" not in result or "task_id" not in result["data"]:
                    raise ValueError("Invalid task ID response")
                return result["data"]["task_id"]
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    http.client.RemoteDisconnected,  # ← 改这里
                    requests.exceptions.HTTPError) as e:
                if attempt < 4:  # not the last attempt
                    wait_time = (2 ** attempt)  # exponential backoff: 1, 2, 4, 8 seconds
                    print(f"[ErnieXVerifier] Get task_id failed (attempt {attempt + 1}/5): {e}, "
                          f"retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[ErnieXVerifier] Get task_id failed after 5 attempts: {e}")
                    raise
            except (ValueError, KeyError) as e:
                print(f"[ErnieXVerifier] Invalid task_id response: {e}")
                raise

    async def submit_reward(self, data):
        """Submit reward request"""
        if self.request_headers:
            result = await http_post(self.submit_url, json=data, headers=self.request_headers, max_retries=3)
        else:
            result = await http_post(self.submit_url, json=data, max_retries=3)
        # print(f"[{data}] submit_reward result is {result}")
        return result

    async def get_reward_result(self, data, max_wait_seconds: int = 300):
        """Get reward result with timeout.

        Args:
            data: Request data
            max_wait_seconds: Maximum time to wait for result (default 5 minutes)
        """
        num_retry = 0
        start_time = time.time()

        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > max_wait_seconds:
                raise TimeoutError(f"Reward result timeout after {max_wait_seconds}s")

            if self.request_headers:
                result = await http_post(self.result_url, json=data, headers=self.request_headers, max_retries=5)
            else:
                result = await http_post(self.result_url, json=data, max_retries=5)

            if result.get("status") == "COMPLETE":
                # print(f"get_reward_result is {result}")
                return result
            else:
                await asyncio.sleep(1 + num_retry)
            num_retry = (num_retry + 1) % 5

    async def _create_default_request(self, data_source, solution_str, ground_truth, extra_info, run_id):
        """
        根据数据协议创建验证器请求数据

        该函数处理不同奖励协议下的数据格式转换，主要包括：
        - normal模式：解析模型生成的思考-回应格式，提取reasoning和answer
        - deep_thinking模式：直接使用原始响应，不做额外解析

        Args:
            data_proto: 数据协议对象，包含模型生成的相关数据
            run_id: 运行标识符，用于请求跟踪

        Returns:
            tuple: (布尔值表示解析是否成功, 奖励请求数据字典, 结果请求数据字典)
                  success: 布尔值，表示数据解析是否成功
                  reward_request: 包含任务配置和验证数据的请求字典
                  result_request: 包含标识信息的简约请求字典
        """
        data_id = "_".join([str(0), str(0)])

        response = solution_str.strip()

        success = False
        if not response.startswith("<think>"):
            response = "<think>" + response
        strict_match = re.search(r"^<think>(.+)</think>(.+)$", response, re.DOTALL)
        if strict_match:
            reasoning, answer = map(str.strip, strict_match.groups())
            success = True
            rsp_match = re.search(r"<response>(.*?)</response>", answer, re.DOTALL)
            if rsp_match:
                answer = rsp_match.group(1)
                if len(answer) > 0 and answer[-1] == "\n":
                    answer = answer[:-1]
                if len(answer) > 0 and answer[0] == "\n":
                    answer = answer[1:]
        else:
            reasoning, answer = None, None
            success = False

        data_verifier = {
            "system": extra_info["meta"].get("system", ""),
            "src": extra_info["meta"].get("raw_src", []),
            "tgt": extra_info["meta"].get("raw_tgt", []),
            "response": answer,
            "verifier": extra_info["verifier"],
        }

        if "reward_shaping" in extra_info["meta"]:
            data_verifier["reward_shaping"] = extra_info["meta"]["reward_shaping"]

        if extra_info["meta"].get("verifier_need_thought", False):
            data_verifier["response"] = response

        reward_request = {
            "task_id": self.task_id,
            "run_id": run_id,
            "data_id": data_id,
            "repeat_id": 0,
            "repeat_count": 1,
            "max_wait_time": 900,
            "need_nl_feedback": True,
            "protocol": self.reward_protocol,
            "data": data_verifier,
        }

        result_request = {"task_id": self.task_id, "items": [{"run_id": run_id, "data_id": data_id}]}
        return success, reward_request, result_request

    # ──────────────────────────────────────────────────────────────────────
    #  Override process() for domain routing (aligned with rl-controller)
    # ──────────────────────────────────────────────────────────────────────
    async def process(self, data_source, solution_str, ground_truth, extra_info, *args, **kwargs):
        """Process data with domain-based routing.

        Routes to different request builders based on data_proto.domain:
        - "code_webdev": webdev tasks with tool_calls
        - "code": code tasks with EB45T/X1/Qwen thinking
        - default: standard reward protocol
        """
        data_id = 0
        gen_id = 0

        step = kwargs.get("step", None)
        run_id = datetime.now().strftime("%S%f") if step is None else str(step)

        try:
            # 1. Deadlock detection (config-controlled)
            if self.need_deadlock_check == 1:
                deadloop = detect_concatenated_repeats(solution_str.strip())
                if isinstance(deadloop, list) and len(deadloop) >= 1:
                    extra_info["meta"]["has_dead_lock"] = True

            # 2. Domain routing for request creation
            success, reward_request, result_request = await self._create_default_request(data_source, solution_str, ground_truth, extra_info, run_id)

            # 3. Success check
            if not success:
                # print(
                #     "When calculating rewards, we fail to get answer content in response, its reward will be 0.",
                # )
                reward = 0
                return reward

            # 4. Max tokens check
            # if "max_tokens" in self._custom_fields and len(data_proto.output_ids) >= self._custom_fields["max_tokens"]:
            #     print(
            #         "The length of output_ids is too long, its reward will be 0.",
            #     )
            #     data_proto.reward = 0
            #     data_proto.reward_finished = True
            #     return data_proto

            # 5. Submit and get reward with retry on failure
            max_submit_retries = 3
            submit_success = False
            last_error = None

            for attempt in range(max_submit_retries):
                try:
                    await self.submit_reward(reward_request)
                    result_res = await self.get_reward_result(result_request)
                    submit_success = True
                    break
                except Exception as e:
                    last_error = e
                    if attempt < max_submit_retries - 1:
                        wait_time = (2 ** attempt)  # 1, 2, 4 seconds
                        print(f"[ErnieXVerifier] Reward request failed (attempt {attempt + 1}/{max_submit_retries}): {e}, "
                              f"retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)

            if not submit_success:
                print(f"[ErnieXVerifier] Reward service unavailable after {max_submit_retries} attempts: {last_error}. "
                      f"Using default_error_reward: {self.default_error_reward}")
                reward = self.default_error_reward
                return reward

            # 6. Extract reward
            reward_num = result_res.get("data", [{}])[0].get("rewards", [self.default_error_reward])[0]
            if reward_num is None:
                # print(
                #     f"reward_num is None, reward set to {self.default_error_reward}",
                # )
                reward_num = self.default_error_reward            
            # print(
            #     f"=====> reward server response: {result_res}",
            # )
            reward = reward_num

        except Exception as e:
            print(f"[ErnieXVerifier] Exception during calculate_reward: {str(e)}, "
                  f"its reward will be {self.default_error_reward}")
            reward = self.default_error_reward

        return reward

    # ──────────────────────────────────────────────────────────────────────
    #  Keep abstract method stubs for BaseVerifier compatibility
    # ──────────────────────────────────────────────────────────────────────

    async def create_request(self, data_proto, *args, **kwargs):
        """Not used — ``process()`` is overridden and handles the full pipeline directly."""
        raise NotImplementedError("process() is overridden — do not call create_request() directly")

    async def get_reward(self, request, data_proto, *args, **kwargs):
        """Not used — ``process()`` is overridden and handles the full pipeline directly."""
        raise NotImplementedError("process() is overridden — do not call get_reward() directly")

    async def calculate_reward(self, response, data_proto, *args, **kwargs) -> float:
        """Not used — ``process()`` is overridden and handles the full pipeline directly."""
        raise NotImplementedError("process() is overridden — do not call calculate_reward() directly")


def compute_score(data_source, solution_str, ground_truth, extra_info=None):

    config = {
        "reward_urls": [
            "http://10.11.153.88:8101/api/v1/reward/task",
            "http://10.11.153.88:8101/api/v1/reward",
            "http://10.11.153.88:8101/api/v1/reward/result"
        ],
        "default_correct_reward": 1.0,
        "default_error_reward": 0.0,
        "reward_client_name": "ernie_x_verifier",
        "reward_auth_key": "dltp_model_online:fabd04e8-c946-4933-8e7b-2d78399d2b03",
        "max_tokens": 40960,
    }
    _verifier = ErnieXVerifier(config)
    reward = asyncio.run(_verifier.process(data_source, solution_str, ground_truth, extra_info))
    print(f"[compute_score DEBUG] reward: {reward}, type: {type(reward)}, data_source: {data_source}, solution_str_len: {len(solution_str)}")
    return reward
