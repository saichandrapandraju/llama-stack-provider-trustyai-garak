"""This contains the implementation of passing garak probes to shields before and after sending prompts to LLM"""

from typing import List, Union
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_stack.apis.safety import RunShieldResponse, ViolationLevel
from llama_stack.apis.inference import OpenAIChatCompletion
from llama_stack_client import LlamaStackClient
import logging

logger = logging.getLogger(__name__)

## from llama_stack/providers/inline/safety/llama_guard/llama_guard.py
CANNED_RESPONSE_TEXT = "I can't answer that. Can I help with something else?"

# Check if GARAK_LOG_FILE environment variable is set
garak_log_file = os.getenv("GARAK_LOG_FILE")
if garak_log_file and not logger.handlers:
    # Create file handler for the garak log file
    file_handler = logging.FileHandler(garak_log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    
    logger.info(f"Shield scan logger configured to write to: {garak_log_file}")


class SimpleShieldOrchestrator:
    def __init__(self) -> None:
        self.llama_stack_client = None
    
    def _get_llama_stack_client(self, base_url: str) -> LlamaStackClient:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        
        base_url = base_url.strip()
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(f"base_url must start with http:// or https://, got: {base_url}")
        
        if self.llama_stack_client is None:
            try:
                self.llama_stack_client = LlamaStackClient(base_url=base_url)
            except Exception as e:
                logger.error(f"Failed to create LlamaStackClient with base_url={base_url}: {e}")
                raise ValueError(f"Failed to initialize LlamaStack client: {e}") from e
        return self.llama_stack_client

    def _get_shield_response(self, shield_id: str, prompt: str, base_url: str, params: dict={}) -> RunShieldResponse:
        if not isinstance(shield_id, str) or not shield_id.strip():
            raise ValueError("shield_id must be a non-empty string")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        
        try:
            messages = [{"role": "user", "content": prompt}]
            return self._get_llama_stack_client(base_url).safety.run_shield(
                messages=messages,
                shield_id=shield_id,
                params=params
            )
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error running shield '{shield_id}': {e}")
            raise e
    
    # # Shall we run shields sequentially and report each shield's response?
    # def _run_shields_with_early_exit(self, shield_ids: List[str], prompt: str, base_url: str, params: dict={}, max_workers: int=5) -> bool:
    #     """Run multiple shields in parallel and return True if any shield returns a violation"""
    #     with ThreadPoolExecutor(max_workers=min(max_workers, len(shield_ids))) as executor:
    #         futures_to_shields = {executor.submit(self._get_shield_response, shield_id, prompt, base_url, params): shield_id for shield_id in shield_ids}
    #         for future in as_completed(futures_to_shields):
    #             shield_id = futures_to_shields[future]
    #             try:
    #                 shield_response = future.result()
    #                 if shield_response.violation and shield_response.violation.violation_level == ViolationLevel.ERROR:
    #                     # cancel pending futures
    #                     for f in list(futures_to_shields.keys()):
    #                         if f != future:  # Don't cancel the current future
    #                             f.cancel()
    #                     return True
    #             except Exception as e:
    #                 logger.error(f"Error running shield {shield_id}: {e}")
    #                 return False
    #     return False

    def _get_LLM_response(self, model: str, prompt: str, base_url: str, sampling_params: dict={}) -> OpenAIChatCompletion:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        if not isinstance(sampling_params, dict):
            raise ValueError("sampling_params must be a dictionary")
        
        try:
            response = self._get_llama_stack_client(base_url).chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                **sampling_params
            )
            
            # Validate response structure
            if not response or not hasattr(response, 'choices') or not response.choices:
                raise ValueError("LLM returned empty or invalid response")
            if not response.choices[0].message or not hasattr(response.choices[0].message, 'content'):
                raise ValueError("LLM response missing message content")
            
            return response
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error running LLM model '{model}': {e}")
            raise e
    def _is_violation(self, shield_response: RunShieldResponse) -> bool:
        """Check if shield response contains an ERROR-level violation."""
        if not shield_response.violation:
            return False
        return shield_response.violation.violation_level == ViolationLevel.ERROR.value
    
    def _get_violation_message(self, shield_response: RunShieldResponse) -> str:
        """Get violation message or canned response."""
        if not shield_response.violation:
            return CANNED_RESPONSE_TEXT
        return shield_response.violation.user_message if shield_response.violation.user_message else CANNED_RESPONSE_TEXT

    def __call__(self, prompt: str, **kwargs) -> List[Union[str, None]]:
        """
        Orchestrate the shield scan and return the response
        
        Args:
            prompt: The prompt to process
            **kwargs: Required keys: llm_io_shield_mapping, base_url, model
                     Optional keys: params, sampling_params, max_workers
        
        Raises:
            ValueError: If required parameters are missing or invalid
            RuntimeError: If shield or LLM execution fails
        """
        # Validate required parameters
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        if "llm_io_shield_mapping" not in kwargs:
            raise ValueError("llm_io_shield_mapping is required in kwargs")
        if "base_url" not in kwargs:
            raise ValueError("base_url is required in kwargs")
        if "model" not in kwargs:
            raise ValueError("model is required in kwargs")
        
        llm_io_shield_mapping = kwargs["llm_io_shield_mapping"]
        if not isinstance(llm_io_shield_mapping, dict):
            raise ValueError("llm_io_shield_mapping must be a dictionary")
        if "input" not in llm_io_shield_mapping or "output" not in llm_io_shield_mapping:
            raise ValueError("llm_io_shield_mapping must have 'input' and 'output' keys")
        
        input_shields: List[str] = llm_io_shield_mapping["input"]
        output_shields: List[str] = llm_io_shield_mapping["output"]
        
        # Validate shield lists
        if not isinstance(input_shields, list):
            raise ValueError("input shields must be a list")
        if not isinstance(output_shields, list):
            raise ValueError("output shields must be a list")

        if input_shields:
            logger.debug(f"Running input shields: {input_shields}")
            # if self._run_shields_with_early_exit(input_shields, prompt, kwargs["base_url"], kwargs.get("params", {}), max_workers=kwargs.get("max_workers", 5)):
            #     return [CANNED_RESPONSE_TEXT]
            for shield_id in input_shields:
                shield_response = self._get_shield_response(shield_id, prompt, kwargs["base_url"], kwargs.get("params", {}))
                if self._is_violation(shield_response):
                    return [self._get_violation_message(shield_response)]
            logger.debug(f"No input violation detected")
        else:
            logger.debug(f"No input shields detected")
        
        logger.debug(f"Continuing with LLM.")
        model_response = self._get_LLM_response(
            model=kwargs["model"],
            prompt=prompt,
            base_url=kwargs["base_url"],
            sampling_params=kwargs.get("sampling_params", {})
        )
        llm_response: str = model_response.choices[0].message.content

        if output_shields:
            logger.debug(f"Running output shields: {output_shields}")
            # if self._run_shields_with_early_exit(output_shields, llm_response, kwargs["base_url"], kwargs.get("params", {}), max_workers=kwargs.get("max_workers", 5)):
            #     return [CANNED_RESPONSE_TEXT]
            for shield_id in output_shields:
                shield_response = self._get_shield_response(shield_id, llm_response, kwargs["base_url"], kwargs.get("params", {}))
                if self._is_violation(shield_response):
                    return [self._get_violation_message(shield_response)]
            logger.debug(f"No output violation detected. Returning LLM response.")
        else:
            logger.debug(f"No output shields detected. Returning LLM response.")
        
        return [llm_response]

    def close(self):
        """Close client for current process"""
        if self.llama_stack_client:
            self.llama_stack_client.close()
            self.llama_stack_client = None
    

simple_shield_orchestrator = SimpleShieldOrchestrator()