"""Streamlit app for AUTOSAR RAG with document retrieval and chat interface."""

from typing import List
import re
import debugpy
import streamlit as st
from phi.document import Document
from phi.utils.log import logger
from PIL import Image
from docparser.docparser import pdf_parser
from assistant.assistant import get_rag_assistant
from posthdl.puml2img import generate_plantuml_image

autosar_basic_puml = """
@startuml
/'AUTOSAR SW layers include: 
- System Services
- Onboard Device Abstraction
- Microcontroller Driver
- Memory Services
- Memory Hardware Abstraction
- Memory Drivers
- Communication Services
- Communication Hardware Abstraction
- Communication Drivers
- I/O Hardware Abstraction
- I/O Drivers
- Complex Drivers'/

/'ASW: application software layer'/
rectangle "ASW" #gray {
[SWC1]
[SWC2]
}

/'RTE: run-time environment layer'/
rectangle "RTE" #gray

/'BSW: basic software layer'/
rectangle "BSW" {
    rectangle "Stack System" {
        rectangle "System Services" #MediumPurple {
            [AUTOSAR OS] /'AUTOSAR operationing system'/
            [Dem] /'Diagnostic Event Manager'/
            [EcuM] /'ECU State Manager'/
            [FiM] /'Function Inhibition Manager'/
            [Det] /'Default Error Tracer'/
            [Dlt] /'Diagnostic Log and Trace'/
            [Csm] /'Crypto Service Manager'/
            [StbM] /'Synchronized Time-Base Manager'/
            [Tm] /'Time Service'/
            [WdgM] /'Watchdog Manager'/
            [ComM] /'COM Manager'/
            [BswM] /'BSW Mode Manager'/
        }
        rectangle "Onboard Device Abstraction" #YellowGreen {
            [WdgIf] /'Watchdog Interface'/
        }
        rectangle "Microcontroller Driver" #Pink {
            [Gpt] /'General Purpose Timer Driver'/
            [Wdg] /'Watchdog Driver'/
            [Mcu] /'Microcontroller Driver'/
            [CorTst] /'Core Test'/
        }

        "System Services" -[hidden]> "Onboard Device Abstraction"
        "System Services" -[hidden]> "Microcontroller Driver"
        "Onboard Device Abstraction" --[hidden]> "Microcontroller Driver"
    }

    "Stack System" -[hidden]> "Memory Stack"

    rectangle "Memory Stack" {
        rectangle "Memory Services" #MediumPurple {
            [NvM] /'NVRAM Manager'/
        }
        rectangle "Memory Hardware Abstraction" #YellowGreen {
            [MemIf] /'Memory Abstraction Interface'/
            [Ea] /'EEPROM Abstraction'/
            [Fee] /'Flash EEPROM Emulation'/
        }    
        rectangle "Memory Drivers" #Pink {
            [FlsTst] /'Flash Test'/
            [RamTst] /'RAM Test'/
            [Fls] /'Flash Driver'/
            [Eep] /'EEPROM Driver'/
        }

        "Memory Services" --[hidden]> "Memory Hardware Abstraction"
        "Memory Hardware Abstraction" ---[hidden]> "Memory Drivers"
    }

    rectangle "Com Stack" {
        rectangle "Communication Services" #MediumPurple {
            [Com] /'Communication'/
            [Dcm] /'Diagnostic Communication Manager'/
            [Dbg] /'debug'/
            [PduR] /'PDU Router'/
            [IpduM] /'IPDU Multiplexer'/
            [SecOC] /'Secure Onboard Communication'/
            [Xf] /'Transformer'/
            [NmIf] /'Network Management Interface'/
            [SM] /'State Manager'/
            [Nm] /'Network Management'/
            [Tp] /'Transport Layer'/
        }
        rectangle "Communication Hardware Abstraction" #YellowGreen {
            [xxx Interface]
            [Trcv] /'Tranceiver Driver'/
            [ext Drv] /'external driver'/
        }
        rectangle "Communication Drivers" #Pink {
            [Spi] /'SPI Handler Driver'/
            [Can] /'CAN Driver'/
            [Lin] /'LIN Driver'/
            [Eth] /'Ethernet Driver'/
            [Fr] /'FlexRay Driver'/
        }

        "Communication Services" ----[hidden]> "Communication Hardware Abstraction"
        "Communication Hardware Abstraction" ---[hidden]> "Communication Drivers"
    }

    "Memory Stack" -[hidden]> "Com Stack"

    rectangle "I/O Stack" {
        rectangle "I/O Hardware Abstraction" #YellowGreen {
            [I/O Signal Interface]
            [Driver for external ADC ASIC]
            [Driver for external I/O ASIC]
        }
        rectangle "I/O Drivers" #Pink {
            [Ocu] /'Output Compare Driver'/
            [Icu] /'Input Capture Unit Driver'/
            [Pwm] /'PWM Driver'/
            [Adc] /'ADC Driver'/
            [Dio] /'Digital Input/Output Driver'/
            [Port] /'Port Driver'/
        }

        "I/O Hardware Abstraction" ---[hidden]> "I/O Drivers"
    }

    "Com Stack" --[hidden]> "I/O Stack"
  
    rectangle "Complex Drivers" {
        [Cdd_1]
    }

    "I/O Stack" -[hidden]> "Complex Drivers"
}

rectangle "MCAL" #gray /'Microcontroller Abstraction Layer'/

ASW --[hidden]> RTE
RTE --[hidden]> BSW
BSW --------[hidden]> MCAL

"Communication Hardware Abstraction" -[hidden]> "I/O Hardware Abstraction"
"Communication Drivers" -[hidden]> "I/O Drivers"

@enduml
"""

def extract_plantuml_code(response):
    """从响应中提取 PlantUML 代码"""
    pattern = r'@startuml[\s\S]*?@enduml'
    match = re.search(pattern, response)
    if match:
        return match.group(0)
    return None

def restart_assistant():
    """Restart assistant."""
    st.session_state["rag_assistant"] = None
    st.session_state["rag_assistant_run_id"] = None
    if "url_scrape_key" in st.session_state:
        st.session_state["url_scrape_key"] += 1
    if "file_uploader_key" in st.session_state:
        st.session_state["file_uploader_key"] += 1
    st.rerun()

def update_llm_config(llm_model, embeddings_model):
    """Check llm config."""
    # Set assistant_type in session state
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = llm_model
    # Restart the assistant if assistant_type has changed
    elif st.session_state["llm_model"] != llm_model:
        st.session_state["llm_model"] = llm_model
        restart_assistant()

    if "embeddings_model" not in st.session_state:
        st.session_state["embeddings_model"] = embeddings_model
    # Restart the assistant if assistant_type has changed
    elif st.session_state["embeddings_model"] != embeddings_model:
        st.session_state["embeddings_model"] = embeddings_model
        st.session_state["embeddings_model_updated"] = True
        restart_assistant()

def get_or_create_assistant(llm_model, embeddings_model):
    """Get or create assistant."""
    if "rag_assistant" not in st.session_state or st.session_state["rag_assistant"] is None:
        logger.info("---*--- Creating %s Assistant ---*---", llm_model)
        rag_assistant = get_rag_assistant(llm_model=llm_model, embeddings_model=embeddings_model)
        st.session_state["rag_assistant"] = rag_assistant
    else:
        rag_assistant = st.session_state["rag_assistant"]

    try:
        st.session_state["rag_assistant_run_id"] = rag_assistant.create_run()
    except (ConnectionError, TimeoutError) as e:
        st.warning(f"Could not create assistant: {str(e)}. Is the database running?")
        return None

    return rag_assistant

def load_assistant_chat_history(rag_assistant):
    """Load chat history from the assistant's memory."""
    assistant_chat_history = rag_assistant.memory.get_chat_history()
    if len(assistant_chat_history) > 0:
        logger.debug("Loading chat history")
        st.session_state["messages"] = assistant_chat_history
    else:
        logger.debug("No chat history found")
        st.session_state["messages"] = [{
            "role": "assistant", 
            "content": "Upload a doc or ask me questions about AUTOSAR directly..."
        }]

def load_knowledge_base(rag_assistant):
    """Load knowledge base."""
    if not rag_assistant.knowledge_base:
        return

    # Add PDFs to knowledge base
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 100

    uploaded_file = st.sidebar.file_uploader(
        "Add a PDF :page_facing_up:", type="pdf", key=st.session_state["file_uploader_key"]
    )
    if uploaded_file is not None:
        process_uploaded_file(rag_assistant, uploaded_file)

    if rag_assistant.knowledge_base.vector_db:
        if st.sidebar.button("Clear Knowledge Base", help="Clear all documents from the knowledge base."):
            rag_assistant.knowledge_base.vector_db.clear()
            st.sidebar.success("Knowledge base cleared")

def process_uploaded_file(rag_assistant, uploaded_file):
    """Process the uploaded PDF file."""
    alert = st.sidebar.info("Processing PDF...", icon="🧠")
    rag_name = uploaded_file.name.split(".")[0]
    if f"{rag_name}_uploaded" not in st.session_state:
        # chunk_size and separators can be set here
        rag_documents: List[Document] = pdf_parser(uploaded_file, chunk_size=300)
        if rag_documents:
            rag_assistant.knowledge_base.load_documents(rag_documents, upsert=True)
        else:
            st.sidebar.error("Could not read PDF")
        st.session_state[f"{rag_name}_uploaded"] = True
    alert.empty()

def initialize_debugger(enable_debug: bool = False):
    """Initialize debugpy for remote debugging if enabled."""
    if enable_debug and "debugpy_initialized" not in st.session_state:
        st.session_state.debugpy_initialized = True
        debugpy.listen(("localhost", 5678))
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()

def set_page_config():
    """Set page config."""
    st.image("assets/robot_autosar.jpeg", width=700)  # Adjust the width as needed
    st.markdown("## :oncoming_automobile: Local AUTOSAR Assistant")
    st.markdown("Welcome to the Local AUTOSAR Assistant. This tool helps you with AUTOSAR-related tasks.")
    st.markdown("Github: [autosar-rag](https://github.com/yyxxrr739/autosar-rag)")
    st.markdown("---")

def update_session_content(session_messages, rag_assistant):
    """Update session content."""
    # Prompt for user input using a chat input box from Streamlit
    if prompt := st.chat_input():
        session_messages.append({"role": "user", "content": prompt})

    # Display existing chat messages
    for message in session_messages:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is from a user, generate a new response
    last_message = session_messages[-1]
    if last_message.get("role") == "user":
        question = last_message["content"]
        if "/bd" in question:
            question = question.replace("/bd", "").strip()
            question += " \n Generate corresponding plantUML code(marked with startuml and enduml) of block diagram based on the following plantUML code of high level AUTOSAR architecture block digram: \n"
            question += autosar_basic_puml
            with st.chat_message("assistant"):
                response = ""
                resp_container = st.empty()
                for delta in rag_assistant.run(question):
                    response += delta  # type: ignore
                    resp_container.markdown(response)
                session_messages.append({"role": "assistant", "content": response})

                # Check if the response contains PlantUML code and export it
                plantuml_code = extract_plantuml_code(response)
                if plantuml_code:
                    with open("generated_diagram.puml", "w") as file:
                        file.write(plantuml_code)

                # Add the generated plantUML image and display it in a new popup window
                image_path = "output.png"
                generate_plantuml_image("generated_diagram.puml", image_path)
                image = Image.open(image_path)
                st.session_state["messages"].append({"role": "assistant", "content_type": "image", "content": image})
                st.image(image, caption="Generated AUTOSAR PlantUML Diagram")

        else:
            with st.chat_message("assistant"):
                response = ""
                resp_container = st.empty()
                for delta in rag_assistant.run(question):
                    response += delta  # type: ignore
                    resp_container.markdown(response)
                session_messages.append({"role": "assistant", "content": response})

def load_assistant_storage(rag_assistant, llm_model, embeddings_model):
    """Load assistant storage."""
    if rag_assistant.storage:
        rag_assistant_run_ids: List[str] = rag_assistant.storage.get_all_run_ids()
    new_rag_assistant_run_id = st.sidebar.selectbox("Run ID", options=rag_assistant_run_ids)
    if st.session_state["rag_assistant_run_id"] != new_rag_assistant_run_id:
        logger.info("---*--- Loading %s run: %s ---*---", llm_model, new_rag_assistant_run_id)
        st.session_state["rag_assistant"] = get_rag_assistant(
            llm_model=llm_model, embeddings_model=embeddings_model, run_id=new_rag_assistant_run_id
        )
        st.rerun()

def main(enable_debug: bool = False) -> None:
    """Main function for the Streamlit app."""
    # Set page config
    set_page_config()

    # Start debugger on localhost:5678，and ensure listen only once
    initialize_debugger(enable_debug)

    # Side bar configuration for llm and embeddings
    llm_model = st.sidebar.selectbox(
        "Select Model", 
        options=["llama3.2", "llama3.2:1b", "llama3.1", "llama3", "phi3", "openhermes", "llama2","gpt-3.5-turbo"]
    )
    embeddings_model = st.sidebar.selectbox(
        "Select Embeddings",
        options=["nomic-embed-text", "llama3", "openhermes", "phi3"],
        help="When you change the embeddings model, the documents will need to be added again.",
    )

    update_llm_config(llm_model, embeddings_model)

    # create instant of assistant
    rag_assistant = get_or_create_assistant(llm_model, embeddings_model)

    load_assistant_chat_history(rag_assistant)

    # update session content with user input and assistant response
    session_messages = st.session_state["messages"]
    update_session_content(session_messages, rag_assistant)

    load_knowledge_base(rag_assistant)

    load_assistant_storage(rag_assistant, llm_model, embeddings_model)

    if st.sidebar.button(
        "New Run",
        help="Reset the session and reload the assistant."
    ):
        restart_assistant()

if __name__ == "__main__":
    main(enable_debug=False)
