import os
import asyncio
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are OSFI Navigator, an expert AI assistant specializing in Canadian financial regulatory compliance.

Answer questions using ONLY the provided regulatory documents.

RULES:
1. Base answers EXCLUSIVELY on the provided context
2. Always cite the specific regulatory source and section
3. If not in context, say: "I cannot find information about this in the regulatory documents provided."
4. Never invent regulatory requirements

Context:
{context}

Question: {question}"""

def get_builtin_corpus():
    return [
        Document(page_content="OSFI Guideline E-23 Model Risk Management September 2025. Applies to all FRFIs effective May 1 2027. THREE OUTCOMES: 1. Model risk is well understood and managed. 2. Model risk is managed using a risk-based approach. 3. Model governance covers the entire model lifecycle. FIVE DATA PROPERTIES: Accurate and fit-for-use, Relevant and representative, Compliant, Traceable, Timely. QUALITATIVE RISK FACTORS: level of autonomy, complexity, customer impacts, regulatory risk. INDEPENDENT REVIEW required before initial deployment and for material changes.", metadata={"title": "OSFI Guideline E-23", "section": "Model Risk Management", "domain": "osfi_e23"}),
        Document(page_content="OSFI Guideline B-20 Residential Mortgage Underwriting. MINIMUM QUALIFYING RATE MQR: greater of contract rate plus 200 basis points or floor of 5.25 percent. STRAIGHT SWITCH EXEMPTION November 2024: same-lender same-amount same-amortization renewals exempt from stress test. Switching lenders at renewal requires re-qualifying. GDS LIMIT 39 percent. TDS LIMIT 44 percent. LTV LIMIT 80 percent for uninsured mortgages.", metadata={"title": "OSFI Guideline B-20", "section": "Mortgage Underwriting", "domain": "b20"}),
        Document(page_content="FINTRAC PCMLTFA Anti-Money Laundering. LARGE CASH TRANSACTION REPORT LCTR: file when receiving CAD 10000 or more in cash in single transaction or within 24 hours by same person. SUSPICIOUS TRANSACTION REPORT STR: file as soon as practicable when reasonable grounds to suspect ML or TF. STRUCTURING: deliberately breaking transactions below LCTR threshold is criminal offence under PCMLTFA section 463. BENEFICIAL OWNERSHIP: must identify all owners at 25 percent or greater threshold.", metadata={"title": "FINTRAC / PCMLTFA", "section": "AML/KYC Requirements", "domain": "fintrac"}),
        Document(page_content="IFRS 9 Expected Credit Loss ECL. Stage 1 performing no SICR recognize 12-month ECL. Stage 2 significant increase in credit risk recognize lifetime ECL. Stage 3 credit-impaired recognize lifetime ECL. SICR rebuttable presumption if more than 30 days past due. MANAGEMENT OVERLAYS permitted with documented evidence and senior governance approval.", metadata={"title": "IFRS 9 ECL", "section": "Expected Credit Loss", "domain": "ifrs9"}),
        Document(page_content="Basel III OSFI Capital Adequacy Requirements CAR 2026. OUTPUT FLOOR kept at 67.5 percent until further notice per February 2025 OSFI announcement. CET1 MINIMUM for D-SIBs: 4.5 plus 2.5 conservation buffer plus 1.0 D-SIB surcharge equals 8.0 percent. LCR must be 100 percent or more. NSFR must be 100 percent or more.", metadata={"title": "Basel III / OSFI CAR", "section": "Capital Adequacy", "domain": "basel3"}),
        Document(page_content="PIPEDA and Quebec Law 25 Privacy for AI Systems. PIPEDA requires meaningful consent for collection use and disclosure of personal information. QUEBEC LAW 25 SECTION 12.1: when decision made exclusively by automated means must inform person, allow observations to human, provide access to personal information used. In force September 2023. BILL C-27 AIDA died January 2025.", metadata={"title": "PIPEDA / Quebec Law 25", "section": "Privacy for AI Systems", "domain": "pipeda"}),
        Document(page_content="CASL Canada Anti-Spam Legislation. CEMs require prior express or implied consent, sender identification, and unsubscribe within 10 business days. EXPRESS CONSENT requires positive deliberate action. Pre-checked boxes do NOT constitute valid express consent. IMPLIED CONSENT expires after 2 years for business relationships. PENALTIES up to CAD 10000000 per violation for organizations.", metadata={"title": "CASL", "section": "Anti-Spam Requirements", "domain": "casl"}),
    ]

_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
        docs = get_builtin_corpus()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        _vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
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
