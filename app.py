import streamlit as st
import os
import re
import time
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults
from markdown_pdf import MarkdownPdf, Section

# Loading the environment variables
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in environment variables.")
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY not found in environment variables.")

# Initializing the LLM
llm = ChatGroq(model="mistral-saba-24b", temperature=0, streaming=True)

#
def retriever_tool(query):
    return "Retrieved data on AI in healthcare: Recent advancements include AI-driven diagnostics and personalized medicine based on latest studies."

retriever_tool = Tool.from_function(
    func=retriever_tool,
    name="FAISS Retriever",
    description="Retrieve relevant documents from a local knowledge base on AI in healthcare."
)

tavily_tool = TavilySearchResults()

# Setting up the Streamlit app
st.set_page_config(page_title="AI in Healthcare Summarization App", page_icon="🩺")
st.title("🩺 AI in Healthcare Summarization App")
st.markdown("This app summarizes the latest trends in AI for healthcare and generates a downloadable report.")
st.write("Current Working Directory (where files are saved):", os.getcwd())

# Initializing the multi agents (researcher, writer, critic, report generator)
researcher_prompt = hub.pull("hwchase17/react")
researcher_agent = create_react_agent(llm=llm, tools=[retriever_tool, tavily_tool], prompt=researcher_prompt)
researcher_agent_executor = AgentExecutor(agent=researcher_agent, tools=[retriever_tool, tavily_tool], verbose=True)

writer_prompt = hub.pull("hwchase17/react")
writer_agent = create_react_agent(llm=llm, tools=[], prompt=writer_prompt)
writer_agent_executor = AgentExecutor(agent=writer_agent, tools=[], verbose=True, handle_parsing_errors=True)

critic_prompt = hub.pull("hwchase17/react")
critic_agent = create_react_agent(llm=llm, tools=[tavily_tool], prompt=critic_prompt)
critic_agent_executor = AgentExecutor(agent=critic_agent, tools=[tavily_tool], verbose=True, handle_parsing_errors=True)

#Function to generate the report
def generate_report_tool(content: str, output_md_path: str = None, output_pdf_path: str = None):
    timestamp = int(time.time())
    if output_md_path is None:
        output_md_path = f"report_{timestamp}.md"
    if output_pdf_path is None:
        output_pdf_path = f"report_{timestamp}.pdf"

    original_summary = ""
    critique = ""
    revised_summary = ""
    if "Original Summary:" in content and "Critique:" in content and "Revised Summary:" in content:
        parts = content.split("Original Summary:")[1].split("Critique:")
        original_summary = parts[0].strip()
        remaining = parts[1].split("Revised Summary:")
        critique = remaining[0].strip()
        revised_summary = remaining[1].strip()
    else:
        revised_summary = content

    markdown_content = f"""# AI in Healthcare Report
## Summary of Latest Trends and Insights (Original)
{original_summary}
## Critique and Suggested Improvements
{critique}
## Summary of Latest Trends and Insights (Revised)
{revised_summary}
## Conclusion
This report summarizes the latest developments in AI for healthcare based on recent data and news.
"""
    try:
        with open(output_md_path, 'w', encoding='utf-8') as md_file:
            md_file.write(markdown_content)
        pdf = MarkdownPdf(toc_level=2)
        pdf.add_section(Section(markdown_content))
        pdf.meta['title'] = 'AI in Healthcare Report'
        pdf.meta['author'] = 'AI Agent Report Generator Created by Obanla Oluwaseun'
        pdf.save(output_pdf_path)
        return (f"The Report has been generated and saved as {output_md_path} (Markdown) and {output_pdf_path} (PDF)", output_md_path, output_pdf_path)
    except Exception as e:
        error_msg = f"Error saving files: {str(e)}. Attempted paths: {output_md_path}, {output_pdf_path}"
        print(error_msg)  # Log to console for debugging
        return (error_msg, None, None)

report_tool = Tool.from_function(
    func=generate_report_tool,
    name="ReportGeneratorTool",
    description="Generate a structured report in Markdown and PDF format from provided content."
)
report_prompt = hub.pull("hwchase17/react")
report_agent = create_react_agent(llm=llm, tools=[report_tool], prompt=report_prompt)
report_agent_executor = AgentExecutor(agent=report_agent, tools=[report_tool], verbose=True)

# Initializing the session state for tracking
if 'stage' not in st.session_state:
    st.session_state.stage = "input"
if 'md_path' not in st.session_state:
    st.session_state.md_path = None
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None

# Adding a reset button to handle unexpected state issues
if st.button("Reset Workflow"):
    st.session_state.stage = "input"
    st.session_state.md_path = None
    st.session_state.pdf_path = None
    st.experimental_rerun()

with st.form("summarization_form", clear_on_submit=False):
    user_query = st.text_input("Enter your query on AI in Healthcare (or use default):",
                               value="Summarize the latest news on AI in Healthcare, including detailed information on specific initiatives, events, and technologies.")
    submitted = st.form_submit_button("Generate Summary and Report")

if submitted or st.session_state.stage != "input":
    if st.session_state.stage == "input":
        st.session_state.stage = "research"
        st.session_state.user_query = user_query

    # The Research Agent Stage
    if st.session_state.stage == "research":
        with st.container():
            st.header("Researcher Agent Output")
            st_callback_research = StreamlitCallbackHandler(st.container())
            with st.spinner("Researching latest AI in Healthcare trends..."):
                try:
                    research_result = researcher_agent_executor.invoke(
                        {"input": st.session_state.user_query},
                        {"callbacks": [st_callback_research]}
                    )
                    st.session_state.research_output = research_result['output']
                    st.write("Research Output:", st.session_state.research_output)
                except Exception as e:
                    st.error(f"Error during research stage: {str(e)}")
                    st.session_state.research_output = "Error occurred during research."
        st.session_state.stage = "write"

    # The Writer agent Stage
    if st.session_state.stage == "write":
        with st.container():
            st.header("Writer Agent Output")
            st_callback_write = StreamlitCallbackHandler(st.container())
            with st.spinner("Writing initial summary..."):
                try:
                    write_result = writer_agent_executor.invoke(
                        {"input": f"Generate a detailed summary based on this data: {st.session_state.research_output}"},
                        {"callbacks": [st_callback_write]}
                    )
                    st.session_state.write_output = write_result['output']
                    st.write("Original Summary:", st.session_state.write_output)
                except Exception as e:
                    st.error(f"Error during writing stage: {str(e)}")
                    st.session_state.write_output = "Error occurred during writing."
        st.session_state.stage = "critique"

    # The Critique agent stage
    if st.session_state.stage == "critique":
        with st.container():
            st.header("Critic Agent Output")
            st_callback_critique = StreamlitCallbackHandler(st.container())
            with st.spinner("Critiquing the summary..."):
                try:
                    critique_result = critic_agent_executor.invoke(
                        {"input": f"Review this summary for accuracy and completeness, suggesting specific improvements: {st.session_state.write_output}"},
                        {"callbacks": [st_callback_critique]}
                    )
                    st.session_state.critique_text = critique_result['output']
                    st.write("Critique Suggestions:", st.session_state.critique_text)
                except Exception as e:
                    st.error(f"Error during critique stage: {str(e)}")
                    st.session_state.critique_text = "Error occurred during critique."
        st.session_state.stage = "revise"

    # Revision Loop to ensure that the write agent takes the corrections of the critique agent and implements it
    if st.session_state.stage == "revise":
        max_revision_attempts = 2
        revision_count = 0
        current_summary = st.session_state.write_output
        current_critique = st.session_state.critique_text

        with st.container():
            st.header("Revision Process")
            while revision_count < max_revision_attempts:
                if any(keyword in current_critique.lower() for keyword in ["improve", "add", "more detail", "specific", "lacking", "missing"]):
                    st.write(f"Revision {revision_count + 1}/{max_revision_attempts}: Critic feedback suggests improvements.")
                    st_callback_revise = StreamlitCallbackHandler(st.container())
                    with st.spinner(f"Revising summary (Iteration {revision_count + 1})..."):
                        try:
                            revised_write_result = writer_agent_executor.invoke(
                                {"input": f"Revise this summary based on the following critique: Summary: {current_summary} | Critique: {current_critique}"},
                                {"callbacks": [st_callback_revise]}
                            )
                            current_summary = revised_write_result['output']
                            st.write(f"Revised Summary (Iteration {revision_count + 1}):", current_summary)
                        except Exception as e:
                            st.error(f"Error during revision (Iteration {revision_count + 1}): {str(e)}")
                            break

                    st_callback_critique_new = StreamlitCallbackHandler(st.container())
                    with st.spinner(f"Re-evaluating revised summary (Iteration {revision_count + 1})..."):
                        try:
                            critique_result = critic_agent_executor.invoke(
                                {"input": f"Review this revised summary for accuracy and completeness, suggesting specific improvements: {current_summary}"},
                                {"callbacks": [st_callback_critique_new]}
                            )
                            current_critique = critique_result['output']
                            st.write(f"Updated Critique (Iteration {revision_count + 1}):", current_critique)
                        except Exception as e:
                            st.error(f"Error during re-evaluation (Iteration {revision_count + 1}): {str(e)}")
                            break
                    revision_count += 1
                else:
                    st.write("Critic feedback accepted. No further revisions needed.")
                    break

            if revision_count >= max_revision_attempts:
                st.write(f"Reached maximum revision attempts ({max_revision_attempts}). Proceeding with the latest summary.")

            st.session_state.final_revised_summary = current_summary
        st.session_state.stage = "report"

    # The Report Generation and Download Stage
    if st.session_state.stage == "report":
        with st.container():
            st.header("Final Report")
            st_callback_report = StreamlitCallbackHandler(st.container())
            with st.spinner("Generating final report..."):
                try:
                    report_query = f"Generate a structured report with these components: Original Summary: {st.session_state.write_output} | Critique: {st.session_state.critique_text} | Revised Summary: {st.session_state.final_revised_summary}"
                    report_result = report_agent_executor.invoke(
                        {"input": report_query},
                        {"callbacks": [st_callback_report]}
                    )
                    st.write("Report Generation Result:", report_result['output'])
                    result_output = report_result['output']
                    if isinstance(result_output, tuple) and len(result_output) == 3:
                        _, md_path, pdf_path = result_output
                        st.session_state.md_path = md_path
                        st.session_state.pdf_path = pdf_path
                        st.write("Saved Markdown Path:", md_path)
                        st.write("Saved PDF Path:", pdf_path)
                    else:
                        st.error(
                            "Could not extract file paths from report generation result. Check console for details.")
                        st.write("Raw Output for Debugging:", result_output)
                        # Fallback to regex as a last resort
                        md_match = re.search(r'(report_\d+\.md)', str(result_output))
                        pdf_match = re.search(r'(report_\d+\.pdf)', str(result_output))
                        if md_match and pdf_match:
                            st.session_state.md_path = md_match.group(1)
                            st.session_state.pdf_path = pdf_match.group(1)
                            st.write("Fallback Saved Markdown Path:", st.session_state.md_path)
                            st.write("Fallback Saved PDF Path:", st.session_state.pdf_path)
                        else:
                            st.session_state.md_path = None
                            st.session_state.pdf_path = None
                            st.error("Regex fallback failed. No file paths found in output.")
                except Exception as e:
                    st.error(f"Error during report generation: {str(e)}")
                    st.session_state.md_path = None
                    st.session_state.pdf_path = None

            # Download section with custom file name (consolidated)
            st.subheader("Download Generated Report")
            custom_pdf_name = st.text_input("Enter a custom name for the PDF file (without extension):",
                                            value="AI_Healthcare_Report")
            custom_pdf_name = custom_pdf_name.strip() + ".pdf" if custom_pdf_name else "AI_Healthcare_Report.pdf"
            custom_md_name = custom_pdf_name.replace(".pdf", ".md")

            col1, col2 = st.columns(2)
            with col1:
                if st.session_state.md_path and os.path.exists(st.session_state.md_path):
                    with open(st.session_state.md_path, "rb") as md_file:
                        st.download_button(
                            label="Download Markdown Report",
                            data=md_file,
                            file_name=custom_md_name,
                            mime="text/markdown",
                            key=f"download_md_{time.time()}"
                        )
                else:
                    st.error("Markdown file not found for download. Check if file was saved correctly.")
            with col2:
                if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
                    with open(st.session_state.pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_file,
                            file_name=custom_pdf_name,
                            mime="application/pdf",
                            key=f"download_pdf_{time.time()}"
                        )
                else:
                    st.error("PDF file not found for download. Check if file was saved correctly.")
        st.session_state.stage = "complete"