# pages/6_LLM_Prompt_Execution.py
import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Generator
import openai
from openai import OpenAI

from modules.nav import Navbar

# Initialize session state
if 'generated_prompts' not in st.session_state:
    st.session_state.generated_prompts = None
if 'prompt_responses' not in st.session_state:
    st.session_state.prompt_responses = {}
if 'execution_history' not in st.session_state:
    st.session_state.execution_history = []
if 'streaming_outputs' not in st.session_state:
    st.session_state.streaming_outputs = {}
if 'expand_all_clusters' not in st.session_state:
    st.session_state.expand_all_clusters = True  # Default to expanded


class OpenAICompatibleClient:
    """Client for communicating with OpenAI-compatible endpoints (like llama.cpp server)"""

    def __init__(self, base_url: str = "http://localhost:8888", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or "dummy-key"  # OpenAI client requires a key
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    def chat_completion_stream(self, messages: List[Dict[str, str]],
                              model: str = "default",
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None) -> Generator[str, None, None]:
        """
        Stream chat completion response from OpenAI-compatible endpoint

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate

        Yields:
            Chunks of response text
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Error during streaming: {str(e)}"

    def chat_completion(self, messages: List[Dict[str, str]],
                       model: str = "default",
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Send chat completion request to OpenAI-compatible endpoint

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Response from the API
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.model_dump()
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}

    def completion(self, prompt: str,
                   model: str = "default",
                   temperature: float = 0.7,
                   max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Send completion request to OpenAI-compatible endpoint

        Args:
            prompt: The prompt text
            model: Model name to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Response from the API
        """
        try:
            response = self.client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.model_dump()
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}


def execute_prompt(client: OpenAICompatibleClient,
                  prompt: str,
                  model: str = "default",
                  temperature: float = 0.7,
                  max_tokens: Optional[int] = None,
                  use_chat_format: bool = True,
                  stream: bool = False) -> Dict[str, Any]:
    """
    Execute a single prompt and return the response

    Args:
        client: OpenAI-compatible client
        prompt: Prompt text to execute
        model: Model name to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        use_chat_format: Whether to use chat completions endpoint
        stream: Whether to stream the response

    Returns:
        Dictionary with prompt, response, and metadata
    """
    start_time = datetime.now()

    if use_chat_format:
        messages = [{"role": "user", "content": prompt}]

        if stream:
            response_text = ""
            # Return generator for streaming
            return client.chat_completion_stream(messages, model, temperature, max_tokens)
        else:
            response = client.chat_completion(messages, model, temperature, max_tokens)

        # Extract text content from response
        if "choices" in response and len(response["choices"]) > 0:
            response_text = response["choices"][0]["message"]["content"]
        else:
            response_text = str(response.get("error", "No response generated"))
    else:
        response = client.completion(prompt, model, temperature, max_tokens)

        # Extract text content from response
        if "choices" in response and len(response["choices"]) > 0:
            response_text = response["choices"][0]["text"]
        else:
            response_text = str(response.get("error", "No response generated"))

    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()

    return {
        "prompt": prompt,
        "response": response_text,
        "raw_response": response,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "use_chat_format": use_chat_format,
        "execution_time_seconds": execution_time,
        "timestamp": start_time.isoformat(),
        "status": "success" if "error" not in response else "error"
    }


def download_all_responses():
    """Function to download all responses as JSON"""
    if st.session_state.prompt_responses:
        all_data = {
            "execution_metadata": {
                "total_executions": len(st.session_state.prompt_responses),
                "successful_executions": sum(1 for r in st.session_state.prompt_responses.values()
                                           if r.get('status') == 'success'),
                "download_timestamp": datetime.now().isoformat()
            },
            "responses": {}
        }

        for result_key, response in st.session_state.prompt_responses.items():
            # Find the corresponding execution info
            execution_info = next(
                (e for e in st.session_state.execution_history if e['result_key'] == result_key),
                None
            )

            if execution_info:
                response_key = f"cluster_{execution_info['cluster_id']}_prompt_{execution_info['prompt_index']}"
            else:
                response_key = result_key

            all_data["responses"][response_key] = {
                "prompt": response['prompt'],
                "response": response['response'],
                "metadata": {
                    "cluster_id": execution_info['cluster_id'] if execution_info else None,
                    "prompt_index": execution_info['prompt_index'] if execution_info else None,
                    "model": response['model'],
                    "temperature": response['temperature'],
                    "max_tokens": response['max_tokens'],
                    "execution_time_seconds": response['execution_time_seconds'],
                    "timestamp": response['timestamp'],
                    "status": response['status']
                }
            }

        json_str = json.dumps(all_data, indent=2)
        st.download_button(
            label="Download Complete Response JSON",
            data=json_str,
            file_name=f"all_prompt_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="download_all_complete"
        )
    else:
        st.warning("No responses to download yet.")


def main():
    Navbar()

    st.title("🤖 LLM Prompt Execution")
    st.markdown("""
    Execute generated prompts against a local LLM server using OpenAI-compatible endpoints.
    This page allows you to run the prompts generated in Step 3b and collect responses for analysis.
    """)

    # Check if prompts are available
    if not st.session_state.generated_prompts:
        st.warning("⚠️ No prompts found. Please generate prompts first in Step 3b1 - LLM Prompt Generation.")
        st.page_link('pages/5_LLM_Prompt_Generation.py', label='Go to Step 3b1 - LLM Prompt Generation')
        return

    # Sidebar configuration
    st.sidebar.header("⚙️ Execution Configuration")

    # Server configuration
    server_url = st.sidebar.text_input(
        "Llama.cpp local LLM URL",
        value="http://localhost:8888",
        help="Base URL of your llama.cpp server or other OpenAI-compatible endpoint"
    )

    # api_key = st.sidebar.text_input(
    #     "API Key (optional)",
    #     value="",
    #     type="password",
    #     help="API key if your server requires authentication"
    # )

    api_key = None

    # Model and generation parameters
    model_name = st.sidebar.text_input(
        "Model Name",
        value="default",
        help="Model name to use for generation"
    )

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Controls randomness. Lower values make output more deterministic."
    )

    max_tokens = st.sidebar.number_input(
        "Max Tokens (optional)",
        min_value=0,
        max_value=16384,  # Fixed: was max_values
        value=4096,
        step=1024,
        help="Maximum number of tokens to generate. Leave empty for model default."
    )

    endpoint_type = st.sidebar.radio(
        "Endpoint Type",
        options=["v1/completions", "v1/chat/completions"],
        index=1,
        help="Select the endpoint type to use"
    )

    use_chat_format = endpoint_type == "v1/chat/completions"

    # Streaming option (only for chat completions)
    stream_response = False
    if use_chat_format:
        stream_response = st.sidebar.checkbox(
            "Stream Response",
            value=True,
            help="Stream the response as it's generated (only for chat completions)"
        )

    # Initialize client
    client = OpenAICompatibleClient(server_url, api_key if api_key else None)

    # Available Prompts section (full width)
    st.subheader("📝 Available Prompts")

    # Show prompt statistics
    total_prompts = sum(len(prompts) for prompts in st.session_state.generated_prompts.values())
    st.info(f"Total prompts: {total_prompts} across {len(st.session_state.generated_prompts)} clusters")

    # Cluster selection
    cluster_options = ["All"] + [f"Cluster {cid}" for cid in sorted(st.session_state.generated_prompts.keys())]
    selected_cluster = st.selectbox("Select Cluster", cluster_options)

    # Expand/Collapse all toggle
    if st.button("🔽 Collapse All Clusters" if st.session_state.expand_all_clusters else "▶️ Expand All Clusters"):
        st.session_state.expand_all_clusters = not st.session_state.expand_all_clusters
        st.rerun()

    # Display prompts for selected cluster
    if selected_cluster == "All":
        prompts_to_show = {}
        for cluster_id in sorted(st.session_state.generated_prompts.keys()):
            prompts_to_show[cluster_id] = st.session_state.generated_prompts[cluster_id]
    else:
        cluster_id = int(selected_cluster.split(" ")[1])
        prompts_to_show = {cluster_id: st.session_state.generated_prompts[cluster_id]}

    # Create expander for each cluster's prompts
    for cluster_id, cluster_prompts in prompts_to_show.items():
        with st.expander(f"Cluster {cluster_id} ({len(cluster_prompts)} prompts)", expanded=st.session_state.expand_all_clusters):
            for i, prompt_data in enumerate(cluster_prompts):
                prompt_text = prompt_data.get('prompt', '')
                word_count = len(prompt_text.split())

                # Create unique key for this prompt
                prompt_key = f"{cluster_id}_{i}"

                # Display editable prompt with increased height
                edited_prompt = st.text_area(
                    f"Prompt {i+1} ({word_count} words)",
                    prompt_text,
                    height=400,
                    key=f"prompt_text_{cluster_id}_{i}"
                )

                # Update prompt in session state if edited
                if edited_prompt != prompt_text:
                    st.session_state.generated_prompts[cluster_id][i]['prompt'] = edited_prompt
                    prompt_text = edited_prompt

                # Run and Clear buttons
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    if st.button(
                        "▶️ Run",
                        key=f"run_{cluster_id}_{i}",
                        help="Execute this prompt"
                    ):
                        # Clear previous streaming output and response for this prompt
                        if prompt_key in st.session_state.streaming_outputs:
                            del st.session_state.streaming_outputs[prompt_key]

                        # Remove previous execution results for this specific prompt
                        keys_to_remove = []
                        for execution in st.session_state.execution_history:
                            if execution['cluster_id'] == cluster_id and execution['prompt_index'] == i:
                                keys_to_remove.append(execution['result_key'])

                        for key in keys_to_remove:
                            if key in st.session_state.prompt_responses:
                                del st.session_state.prompt_responses[key]

                        # Remove execution history entries for this prompt
                        st.session_state.execution_history = [
                            e for e in st.session_state.execution_history 
                            if not (e['cluster_id'] == cluster_id and e['prompt_index'] == i)
                        ]

                        # Execute the prompt
                        with st.spinner(f"Executing prompt {i+1} from Cluster {cluster_id}..."):
                            result = execute_prompt(
                                client, prompt_text, model_name, temperature,
                                max_tokens if max_tokens > 0 else None, 
                                use_chat_format, stream_response
                            )

                            # Store result in session state with unique key
                            execution_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                            result_key = f"cluster_{cluster_id}_prompt_{i}_{execution_timestamp}"

                            # Handle streaming vs non-streaming response
                            if stream_response and use_chat_format:
                                # For streaming, we need to collect the response and display it
                                # Initialize streaming output in session state
                                prompt_key = f"{cluster_id}_{i}"
                                st.session_state.streaming_outputs[prompt_key] = ""

                                # Create a placeholder for streaming output (full width)
                                streaming_placeholder = st.empty()

                                start_time = datetime.now()
                                response_text = ""

                                # Display streaming response in real-time (full width)
                                for chunk in result:
                                    response_text += chunk
                                    st.session_state.streaming_outputs[prompt_key] = response_text
                                    with streaming_placeholder.container():
                                        st.markdown("### Response")
                                        st.markdown(response_text)

                                end_time = datetime.now()
                                execution_time = (end_time - start_time).total_seconds()

                                # Store final result
                                st.session_state.prompt_responses[result_key] = {
                                    "prompt": prompt_text,
                                    "response": response_text,
                                    "raw_response": {"choices": [{"message": {"content": response_text}}]},
                                    "model": model_name,
                                    "temperature": temperature,
                                    "max_tokens": max_tokens if max_tokens > 0 else None,
                                    "use_chat_format": use_chat_format,
                                    "execution_time_seconds": execution_time,
                                    "timestamp": start_time.isoformat(),
                                    "status": "success"
                                }

                                # Add to execution history
                                st.session_state.execution_history.append({
                                    "cluster_id": cluster_id,
                                    "prompt_index": i,
                                    "result_key": result_key,
                                    "timestamp": datetime.now().isoformat()
                                })

                                # Force a rerun to update the UI
                                st.rerun()
                            else:
                                # Non-streaming response
                                st.session_state.prompt_responses[result_key] = result

                                # Add to execution history
                                st.session_state.execution_history.append({
                                    "cluster_id": cluster_id,
                                    "prompt_index": i,
                                    "result_key": result_key,
                                    "timestamp": datetime.now().isoformat()
                                })

                        st.success(f"✅ Prompt {i+1} executed successfully!")
                        if not (stream_response and use_chat_format):
                            st.rerun()

                with col2:
                    if st.button(
                        "🗑️ Clear Response",
                        key=f"clear_response_{cluster_id}_{i}",
                        help="Clear the response for this prompt"
                    ):
                        # Clear the response for this specific prompt
                        keys_to_remove = []
                        for execution in st.session_state.execution_history:
                            if execution['cluster_id'] == cluster_id and execution['prompt_index'] == i:
                                keys_to_remove.append(execution['result_key'])

                        for key in keys_to_remove:
                            if key in st.session_state.prompt_responses:
                                st.session_state.prompt_responses[key]['response'] = ""
                                st.session_state.prompt_responses[key]['status'] = 'cleared'

                        # Clear streaming output if it exists
                        if prompt_key in st.session_state.streaming_outputs:
                            del st.session_state.streaming_outputs[prompt_key]

                        st.rerun()

                # Display streaming output if it exists for this prompt (full width)
                if prompt_key in st.session_state.streaming_outputs:
                    st.markdown("### Response")
                    st.markdown(st.session_state.streaming_outputs[prompt_key])

                # Display stored response if it exists (after streaming is complete)
                execution_keys = [e['result_key'] for e in st.session_state.execution_history 
                                if e['cluster_id'] == cluster_id and e['prompt_index'] == i]
                if execution_keys:
                    latest_key = execution_keys[-1]
                    if latest_key in st.session_state.prompt_responses:
                        response = st.session_state.prompt_responses[latest_key]
                        if response.get('status') == 'success' and prompt_key not in st.session_state.streaming_outputs:
                            st.markdown("### Response")
                            st.markdown(response['response'])
                        elif response.get('status') == 'cleared':
                            st.info("Response cleared")

    st.divider()

    # Responses section (full width)
    st.subheader("💬 Responses")

    if not st.session_state.prompt_responses:
        st.info("No responses yet. Execute prompts from the section above to see results here.")
    else:
        # Show execution statistics
        successful_executions = sum(1 for r in st.session_state.prompt_responses.values()
                                 if r.get('status') == 'success')
        total_executions = len(st.session_state.prompt_responses)

        col_metrics, col_download_all, col_clear_all = st.columns([3, 1, 1])
        with col_metrics:
            st.metric("Executed Prompts", f"{successful_executions}/{total_executions}")
        with col_download_all:
            if st.button("📦 Download All Responses", key="download_all_top"):
                download_all_responses()
        with col_clear_all:
            if st.button("🧹 Clear All Responses", key="clear_all_top"):
                # Clear all responses
                for key in st.session_state.prompt_responses:
                    st.session_state.prompt_responses[key]['response'] = ""
                    st.session_state.prompt_responses[key]['status'] = 'cleared'

                # Clear all streaming outputs
                st.session_state.streaming_outputs.clear()
                st.rerun()

        # Create tabs for different views
        tab1, tab2 = st.tabs(["All Responses", "Execution History"])

        with tab1:
            # Show all responses
            for i, execution in enumerate(reversed(st.session_state.execution_history)):
                result_key = execution['result_key']
                response = st.session_state.prompt_responses[result_key]

                with st.expander(
                    f"Cluster {execution['cluster_id']} - Prompt {execution['prompt_index'] + 1} "
                    f"({response['status']})"
                ):
                    col_download, col_delete, col_clear = st.columns([1, 1, 1])

                    with col_download:
                        if st.button(f"📥 Download", key=f"download_btn_{result_key}"):
                            download_data = {
                                "prompt": response['prompt'],
                                "response": response['response'],
                                "metadata": {
                                    "cluster_id": execution['cluster_id'],
                                    "prompt_index": execution['prompt_index'],
                                    "model": response['model'],
                                    "temperature": response['temperature'],
                                    "max_tokens": response['max_tokens'],
                                    "execution_time_seconds": response['execution_time_seconds'],
                                    "timestamp": response['timestamp']
                                }
                            }

                            json_str = json.dumps(download_data, indent=2)
                            st.download_button(
                                label="Download JSON",
                                data=json_str,
                                file_name=f"prompt_response_{result_key}.json",
                                mime="application/json",
                                key=f"download_widget_{result_key}"
                            )

                    with col_delete:
                        if st.button(f"🗑️ Delete", key=f"delete_{result_key}"):
                            del st.session_state.prompt_responses[result_key]
                            st.session_state.execution_history.remove(execution)
                            st.rerun()

                    with col_clear:
                        if st.button(f"🧹 Clear Response", key=f"clear_{result_key}"):
                            # Clear the response content but keep the prompt
                            response['response'] = ""
                            response['status'] = 'cleared'
                            st.rerun()

                    # Show prompt and response with markdown formatting
                    st.markdown("**Prompt:**")
                    st.markdown(response['prompt'])

                    st.markdown("**Response:**")
                    if response['status'] == 'success':
                        st.markdown(response['response'])
                    elif response['status'] == 'cleared':
                        st.info("Response cleared")
                    else:
                        st.error(f"Error: {response['response']}")

        with tab2:
            # Show execution history as a table
            if st.session_state.execution_history:
                history_data = []
                for execution in st.session_state.execution_history:
                    result_key = execution['result_key']
                    response = st.session_state.prompt_responses[result_key]
                    history_data.append({
                        "Cluster ID": execution['cluster_id'],
                        "Prompt Index": execution['prompt_index'] + 1,
                        "Status": response['status'],
                        "Execution Time (s)": f"{response['execution_time_seconds']:.2f}",
                        "Model": response['model'],
                        "Timestamp": execution['timestamp']
                    })

                df = pd.DataFrame(history_data)
                st.dataframe(df, width='stretch')

    # Bottom section: Bulk operations
    st.divider()
    st.subheader("🔧 Bulk Operations")

    st.write("**Execute All Prompts**")

    # Show confirmation if not confirmed yet
    if not st.session_state.get('confirm_bulk_execution', False):
        if st.button("🚀 Execute All Prompts", type="primary"):
            st.session_state.confirm_bulk_execution = True
            st.rerun()
    else:
        st.warning("⚠️ **Bulk Execution Confirmation**")
        st.write("This will execute all prompts. This may take a long time.")

        col_confirm, col_cancel = st.columns([1, 1])
        with col_confirm:
            if st.button("✅ Yes, Execute All", type="primary"):
                # Proceed with execution
                st.session_state.confirm_bulk_execution = False
                progress_bar = st.progress(0)
                status_text = st.empty()

                total_prompts_to_execute = sum(len(prompts) for prompts in st.session_state.generated_prompts.values())
                executed_count = 0

                for cluster_id, cluster_prompts in st.session_state.generated_prompts.items():
                    for i, prompt_data in enumerate(cluster_prompts):
                        prompt_text = prompt_data.get('prompt', '')

                        status_text.text(f"Executing Cluster {cluster_id}, Prompt {i+1}...")

                        # Execute the prompt
                        result = execute_prompt(
                            client, prompt_text, model_name, temperature,
                            max_tokens if max_tokens > 0 else None,
                            use_chat_format, stream_response
                        )

                        # Store result with unique key
                        execution_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                        result_key = f"cluster_{cluster_id}_prompt_{i}_{execution_timestamp}"

                        # Handle streaming vs non-streaming response
                        if stream_response and use_chat_format:
                            # For streaming, we need to collect the response
                            response_text = ""
                            start_time = datetime.now()

                            # Collect all chunks (no real-time display in bulk)
                            for chunk in result:
                                response_text += chunk

                            end_time = datetime.now()
                            execution_time = (end_time - start_time).total_seconds()

                            # Store final result
                            st.session_state.prompt_responses[result_key] = {
                                "prompt": prompt_text,
                                "response": response_text,
                                "raw_response": {"choices": [{"message": {"content": response_text}}]},
                                "model": model_name,
                                "temperature": temperature,
                                "max_tokens": max_tokens if max_tokens > 0 else None,
                                "use_chat_format": use_chat_format,
                                "execution_time_seconds": execution_time,
                                "timestamp": start_time.isoformat(),
                                "status": "success"
                            }
                        else:
                            # Non-streaming response
                            st.session_state.prompt_responses[result_key] = result

                        st.session_state.execution_history.append({
                            "cluster_id": cluster_id,
                            "prompt_index": i,
                            "result_key": result_key,
                            "timestamp": datetime.now().isoformat()
                        })

                        executed_count += 1
                        progress = executed_count / total_prompts_to_execute
                        progress_bar.progress(progress)

                status_text.text(f"Completed! {executed_count} prompts executed.")
                st.success(f"✅ Bulk execution completed! {executed_count} prompts processed.")
                st.rerun()

        with col_cancel:
            if st.button("❌ Cancel"):
                st.session_state.confirm_bulk_execution = False
                st.rerun()

    # Download All Responses section (below Execute All Prompts)
    st.write("**Download All Responses**")
    if st.session_state.prompt_responses:
        st.info(f"Ready to download {len(st.session_state.prompt_responses)} responses")

        # Create columns for download and clear all buttons
        col_download_all, col_clear_all = st.columns([1, 1])
        with col_download_all:
            download_all_responses()
        with col_clear_all:
            if st.button("🧹 Clear All Responses", key="clear_all_responses"):
                # Clear all responses
                for key in st.session_state.prompt_responses:
                    st.session_state.prompt_responses[key]['response'] = ""
                    st.session_state.prompt_responses[key]['status'] = 'cleared'

                # Clear all streaming outputs
                st.session_state.streaming_outputs.clear()
                st.rerun()
    else:
        st.warning("No responses to download yet. Execute some prompts first.")


if __name__ == '__main__':
    main()