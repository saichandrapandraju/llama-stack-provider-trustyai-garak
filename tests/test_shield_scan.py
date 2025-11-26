"""Tests for shield scanning functionality"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from llama_stack_provider_trustyai_garak.shield_scan import (
    SimpleShieldOrchestrator,
    simple_shield_orchestrator,
    CANNED_RESPONSE_TEXT
)
from llama_stack.apis.safety import RunShieldResponse, ViolationLevel, SafetyViolation


class TestSimpleShieldOrchestrator:
    """Test cases for SimpleShieldOrchestrator"""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator instance"""
        return SimpleShieldOrchestrator()

    @pytest.fixture
    def mock_llama_stack_client(self):
        """Create a mock LlamaStackClient"""
        client = MagicMock()
        return client

    def test_get_llama_stack_client_creates_new(self, orchestrator):
        """Test that _get_llama_stack_client creates a new client"""
        with patch('llama_stack_provider_trustyai_garak.shield_scan.LlamaStackClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            client = orchestrator._get_llama_stack_client("http://test.com")
            
            mock_client_class.assert_called_once_with(base_url="http://test.com")
            assert client == mock_client
            assert orchestrator.llama_stack_client == mock_client

    def test_get_llama_stack_client_reuses_existing(self, orchestrator):
        """Test that _get_llama_stack_client reuses existing client"""
        with patch('llama_stack_provider_trustyai_garak.shield_scan.LlamaStackClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # First call
            client1 = orchestrator._get_llama_stack_client("http://test.com")
            # Second call
            client2 = orchestrator._get_llama_stack_client("http://test.com")
            
            # Should only create once
            mock_client_class.assert_called_once()
            assert client1 == client2

    def test_get_shield_response_success(self, orchestrator):
        """Test successful shield response"""
        mock_client = MagicMock()
        mock_response = RunShieldResponse(violation=None)
        mock_client.safety.run_shield.return_value = mock_response
        
        with patch.object(orchestrator, '_get_llama_stack_client', return_value=mock_client):
            result = orchestrator._get_shield_response(
                shield_id="test_shield",
                prompt="test prompt",
                base_url="http://test.com"
            )
            
            assert isinstance(result, RunShieldResponse)
            assert result.violation is None
            mock_client.safety.run_shield.assert_called_once()

    def test_get_shield_response_with_violation(self, orchestrator):
        """Test shield response with violation"""
        mock_client = MagicMock()
        
        # Create mock response matching production API structure
        mock_violation = Mock()
        mock_violation.violation_level = "error"  # API returns string
        mock_violation.user_message = "Violation detected"
        
        mock_response = Mock()
        mock_response.violation = mock_violation
        
        mock_client.safety.run_shield.return_value = mock_response
        
        with patch.object(orchestrator, '_get_llama_stack_client', return_value=mock_client):
            result = orchestrator._get_shield_response(
                shield_id="test_shield",
                prompt="harmful prompt",
                base_url="http://test.com"
            )
            
            assert result.violation is not None
            assert result.violation.violation_level == "error"

    # def test_run_shields_with_early_exit_no_violation(self, orchestrator):
    #     """Test running shields with no violations"""
    #     mock_response = RunShieldResponse(violation=None)
        
    #     with patch.object(orchestrator, '_get_shield_response', return_value=mock_response):
    #         result = orchestrator._run_shields_with_early_exit(
    #             shield_ids=["shield1", "shield2"],
    #             prompt="test prompt",
    #             base_url="http://test.com"
    #         )
            
    #         assert result is False

    # def test_run_shields_with_early_exit_with_violation(self, orchestrator):
    #     """Test running shields with violation triggers early exit"""
    #     mock_violation = Mock()
    #     mock_violation.violation_level = "error"  # API returns string
    #     mock_violation.user_message = "Violation"
        
    #     mock_response = Mock()
    #     mock_response.violation = mock_violation
        
    #     with patch.object(orchestrator, '_get_shield_response', return_value=mock_response):
    #         result = orchestrator._run_shields_with_early_exit(
    #             shield_ids=["shield1", "shield2"],
    #             prompt="harmful prompt",
    #             base_url="http://test.com"
    #         )
            
    #         assert result is True

    def test_get_llm_response_success(self, orchestrator):
        """Test successful LLM response"""
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="LLM response text"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch.object(orchestrator, '_get_llama_stack_client', return_value=mock_client):
            result = orchestrator._get_LLM_response(
                model="test-model",
                prompt="test prompt",
                base_url="http://test.com"
            )
            
            assert result.choices[0].message.content == "LLM response text"
            mock_client.chat.completions.create.assert_called_once()

    def test_is_violation_true(self, orchestrator):
        """Test _is_violation with error level violation"""
        # Create mock that matches production API response structure
        mock_violation = Mock()
        mock_violation.violation_level = "error"  # API returns string
        mock_violation.user_message = "Violation"
        
        response = Mock()
        response.violation = mock_violation
        
        assert orchestrator._is_violation(response) is True

    def test_is_violation_false_no_violation(self, orchestrator):
        """Test _is_violation with no violation"""
        response = Mock()
        response.violation = None
        assert orchestrator._is_violation(response) is False

    def test_is_violation_false_info_level(self, orchestrator):
        """Test _is_violation with non-error level"""
        mock_violation = Mock()
        mock_violation.violation_level = "info"  # Non-error level
        mock_violation.user_message = "Informational"
        
        response = Mock()
        response.violation = mock_violation
        
        # Should be False since we only check for ERROR level
        assert orchestrator._is_violation(response) is False

    def test_get_violation_message_with_message(self, orchestrator):
        """Test _get_violation_message returns custom message"""
        mock_violation = Mock()
        mock_violation.user_message = "Custom violation message"
        
        response = Mock()
        response.violation = mock_violation
        
        assert orchestrator._get_violation_message(response) == "Custom violation message"

    def test_get_violation_message_without_message(self, orchestrator):
        """Test _get_violation_message returns canned response"""
        mock_violation = Mock()
        mock_violation.user_message = None
        
        response = Mock()
        response.violation = mock_violation
        
        assert orchestrator._get_violation_message(response) == CANNED_RESPONSE_TEXT

    def test_call_with_input_shield_violation(self, orchestrator):
        """Test orchestrator call with input shield violation"""
        # Create mock that matches production API response
        mock_violation = Mock()
        mock_violation.violation_level = "error"  # API returns string
        mock_violation.user_message = "Input violation"
        
        mock_response = Mock()
        mock_response.violation = mock_violation
        
        with patch.object(orchestrator, '_get_shield_response', return_value=mock_response):
            result = orchestrator(
                prompt="harmful prompt",
                llm_io_shield_mapping={"input": ["shield1"], "output": []},
                base_url="http://test.com",
                model="test-model"
            )
            
            assert result == ["Input violation"]

    def test_call_with_output_shield_violation(self, orchestrator):
        """Test orchestrator call with output shield violation"""
        mock_llm_response = Mock()
        mock_llm_response.choices = [Mock(message=Mock(content="harmful response"))]
        
        # Input shield - no violation
        mock_response_safe = Mock()
        mock_response_safe.violation = None
        
        # Output shield - has violation  
        mock_violation = Mock()
        mock_violation.violation_level = "error"  # API returns string
        mock_violation.user_message = "Output violation"
        
        mock_response_violation = Mock()
        mock_response_violation.violation = mock_violation
        
        with patch.object(orchestrator, '_get_LLM_response', return_value=mock_llm_response):
            with patch.object(orchestrator, '_get_shield_response') as mock_get_shield:
                # First call (input shield) - no violation
                # Second call (output shield) - has violation
                mock_get_shield.side_effect = [mock_response_safe, mock_response_violation]
                
                result = orchestrator(
                    prompt="test prompt",
                    llm_io_shield_mapping={"input": ["shield1"], "output": ["shield2"]},
                    base_url="http://test.com",
                    model="test-model"
                )
                
                assert result == ["Output violation"]

    def test_call_no_shields(self, orchestrator):
        """Test orchestrator call with no shields"""
        mock_llm_response = Mock()
        mock_llm_response.choices = [Mock(message=Mock(content="normal response"))]
        
        with patch.object(orchestrator, '_get_LLM_response', return_value=mock_llm_response):
            result = orchestrator(
                prompt="test prompt",
                llm_io_shield_mapping={"input": [], "output": []},
                base_url="http://test.com",
                model="test-model"
            )
            
            assert result == ["normal response"]

    def test_call_no_violation(self, orchestrator):
        """Test orchestrator call with shields but no violations"""
        mock_llm_response = Mock()
        mock_llm_response.choices = [Mock(message=Mock(content="safe response"))]
        mock_shield_response = RunShieldResponse(violation=None)
        
        with patch.object(orchestrator, '_get_shield_response', return_value=mock_shield_response):
            with patch.object(orchestrator, '_get_LLM_response', return_value=mock_llm_response):
                result = orchestrator(
                    prompt="safe prompt",
                    llm_io_shield_mapping={"input": ["shield1"], "output": ["shield2"]},
                    base_url="http://test.com",
                    model="test-model"
                )
                
                assert result == ["safe response"]

    def test_close_client(self, orchestrator):
        """Test closing client"""
        mock_client = Mock()
        orchestrator.llama_stack_client = mock_client
        
        orchestrator.close()
        
        mock_client.close.assert_called_once()
        assert orchestrator.llama_stack_client is None

    def test_close_client_when_none(self, orchestrator):
        """Test closing client when it's None"""
        orchestrator.llama_stack_client = None
        
        # Should not raise an error
        orchestrator.close()
        
        assert orchestrator.llama_stack_client is None


class TestSimpleShieldOrchestratorSingleton:
    """Test the singleton instance"""

    def test_simple_shield_orchestrator_exists(self):
        """Test that simple_shield_orchestrator is available"""
        assert simple_shield_orchestrator is not None
        assert isinstance(simple_shield_orchestrator, SimpleShieldOrchestrator)

    def test_simple_shield_orchestrator_is_singleton(self):
        """Test that the module-level instance is reusable"""
        from llama_stack_provider_trustyai_garak.shield_scan import simple_shield_orchestrator as orch2
        assert simple_shield_orchestrator is orch2
