import os
import asyncio
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

CHROMA_PATH = "./chroma_db"

SYSTEM_PROMPT = """You are OSFI Navigator, an expert AI assistant specializing in Canadian financial regulatory compliance.

You answer questions about Canadian financial regulations using ONLY the provided regulatory documents.

CRITICAL RULES:
1. Base your answer EXCLUSIVELY on the provided context documents
2. Always cite the specific regulatory source and section for each claim
3. If the answer is not in the provided context, say: "I cannot find information about this in the regulatory documents provided. Please consult the primary regulatory source directly."
4. Never invent regulatory requirements
5. Be precise with regulatory language

Context documents:
{context}

Question: {question}"""

def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))

def get_builtin_corpus():
    return [
        Document(page_content="""OSFI Guideline E-23 Model Risk Management September 2025. SCOPE: Applies to all FRFIs effective May 1 2027. THREE OUTCOMES: 1. Model risk is well understood and managed across the enterprise. 2. Model risk is managed using a risk-based approach. 3. Model governance covers the entire model lifecycle. MODEL LIFECYCLE: Design, Review, Deployment, Monitoring, Decommission. FIVE DATA PROPERTIES: Accurate and fit-for-use, Relevant and representative, Compliant, Traceable, Timely. QUALITATIVE RISK FACTORS: level of autonomy, complexity, customer impacts, regulatory risk. QUANTITATIVE RISK FACTORS: portfolio size, financial and operational impacts. INDEPENDENT REVIEW required before initial deployment and for material changes.""", metadata={"title": "OSFI Guideline E-23", "section": "Model Risk Management", "domain": "osfi_e23"}),
        Document(page_content="""OSFI Guideline B-20 Residential Mortgage Underwriting. MINIMUM QUALIFYING RATE MQR: greater of contract rate plus 200 basis points or floor of 5.25 percent. STRAIGHT SWITCH EXEMPTION November 2024: same-lender same-amount same-amortization renewals exempt from stress test. Switching lenders at renewal requires re-qualifying. GDS LIMIT 39 percent. TDS LIMIT 44 percent. Both calculated at MQR stress rate not contract rate. LTV LIMIT 80 percent for uninsured mortgages requiring minimum 20 percent down payment. SELF-EMPLOYED INCOME: bank statements alone insufficient, require NOAs T1 generals business financials.""", metadata={"title": "OSFI Guideline B-20", "section": "Mortgage Underwriting", "domain": "b20"}),
        Document(page_content="""FINTRAC PCMLTFA Anti-Money Laundering. LARGE CASH TRANSACTION REPORT LCTR: file when receiving CAD 10000 or more in cash in single transaction or multiple transactions within 24 hours by same person. SUSPICIOUS TRANSACTION REPORT STR: file as soon as practicable when reasonable grounds to suspect ML or TF. No fixed deadline. STRUCTURING: deliberately breaking transactions below LCTR threshold is criminal offence under PCMLTFA section 463 regardless of whether funds are legitimate. Structuring requires STR filing. BENEFICIAL OWNERSHIP: must identify all owners at 25 percent or greater threshold. POLITICALLY EXPOSED FOREIGN PERSON PEFP: requires senior management approval, source of funds verification, enhanced ongoing monitoring.""", metadata={"title": "FINTRAC / PCMLTFA", "section": "AML/KYC Requirements", "domain": "fintrac"}),
        Document(page_content="""IFRS 9 Expected Credit Loss ECL Canadian Implementation. Adopted by Canadian Big Six banks November 1 2017. THREE STAGES: Stage 1 performing no SICR recognize 12-month ECL. Stage 2 significant increase in credit risk recognize lifetime ECL. Stage 3 credit-impaired recognize lifetime ECL with interest on net carrying amount. SICR significant increase in credit risk: rebuttable presumption if more than 30 days past due. DE-STAGING: when SICR conditions no longer present transfer back to Stage 1 using same forward-looking model criteria. MANAGEMENT OVERLAYS: permitted with documented evidence that model output inadequate, require independent review and senior governance approval.""", metadata={"title": "IFRS 9 ECL", "section": "Expected Credit Loss", "domain": "ifrs9"}),
        Document(page_content="""Basel III OSFI Capital Adequacy Requirements CAR 2026. OUTPUT FLOOR DEFERRAL: February 2025 OSFI deferred increases keeping output floor at 67.5 percent until further notice, minimum two years notice before resuming increases. CET1 MINIMUM for D-SIBs: 4.5 percent minimum plus 2.5 percent conservation buffer plus 1.0 percent D-SIB surcharge equals 8.0 percent effective minimum. Canadian D-SIBs: RBC TD Scotiabank BMO CIBC National Bank. LCR Liquidity Coverage Ratio: HQLA divided by net cash outflows over 30 days must be 100 percent or more. NSFR Net Stable Funding Ratio: available stable funding divided by required stable funding must be 100 percent or more over 1-year horizon.""", metadata={"title": "Basel III / OSFI CAR", "section": "Capital Adequacy", "domain": "basel3"}),
        Document(page_content="""PIPEDA and Quebec Law 25 Privacy for AI Systems. PIPEDA requires meaningful consent for collection use and disclosure of personal information. Secondary use of data beyond original purpose requires renewed consent analysis. QUEBEC LAW 25 SECTION 12.1 automated decision-making: when decision about individual made exclusively by automated means must inform person, allow observations to human, provide access to personal information used. In force September 2023. PRIVACY IMPACT ASSESSMENT PIA: Quebec Law 25 requires PIA before implementing technology projects involving personal information. BILL C-27 AIDA died on order paper January 2025 when Parliament prorogued. Canada has no federal AI statute currently.""", metadata={"title": "PIPEDA / Quebec Law 25", "section": "Privacy for AI Systems", "domain": "pipeda"}),
        Document(page_content="""CASL Canada Anti-Spam Legislation. COMMERCIAL ELECTRONIC MESSAGES CEMs require prior express or implied consent, sender identification, and unsubscribe mechanism processed within 10 business days. EXPRESS CONSENT requires positive deliberate action. Pre-checked boxes do NOT constitute valid express consent. IMPLIED CONSENT expires after 2 years for business relationships or 6 months for inquiries. Inaction not clicking unsubscribe does NOT constitute consent. PENALTIES up to CAD 10000000 per violation for organizations. PROOF OF CONSENT under section 13 sender bears burden of proof. No unsubscribe clicks does not constitute proof of consent.""", metadata={"title": "CASL", "section": "Anti-Spam Requirements", "domain": "casl"}),
    ]

_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = get_embeddings()
        if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
            _vectorstore = Chroma(, embedding_function=embeddings)
        else:
            docs = get_builtin_corpus()
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            _vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, )
    return _vectorstore

async def query_rag(question: str, model: str = "gpt-4o-mini", domain: str = "all") -> dict:
    vectorstore = get_vectorstore()
    search_kwargs = {"k": 5}
    if domain and domain != "all":
        search_kwargs["filter"] = {"domain": domain}
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    loop = asyncio.get_event_loop()
    docs = await loop.run_in_executor(None, retriever.invoke, question)
    if not docs:
        return {"answer": "I cannot find information about this in the regulatory documents provided.", "sources": [], "model_used": model, "grounded": False, "confidence": 0.0}
    context = "\n\n---\n\n".join([f"Source: {d.metadata.get('title')} | Section: {d.metadata.get('section')}\n{d.page_content}" for d in docs])
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT)])
    llm = ChatOpenAI(model=model, temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    chain = prompt | llm
    response = await loop.run_in_executor(None, lambda: chain.invoke({"context": context, "question": question}))
    answer = response.content
    grounded = len(docs) > 0 and "cannot find" not in answer.lower()
    confidence = min(0.95, 0.6 + (len(docs) * 0.07)) if grounded else 0.2
    seen = set()
    sources = []
    for d in docs:
        key = f"{d.metadata.get('title')}:{d.metadata.get('section')}"
        if key not in seen:
            seen.add(key)
            sources.append({"title": d.metadata.get("title", "Unknown"), "section": d.metadata.get("section", "Unknown"), "excerpt": d.page_content[:200] + "...", "relevance": 0.9})
    return {"answer": answer, "sources": sources, "model_used": model, "grounded": grounded, "confidence": round(confidence, 2)}

