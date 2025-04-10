from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from PyPDF2 import PdfReader
import re
from fpdf import FPDF
from io import BytesIO

class PDFComparator:
    def __init__(self, model_path: str):
        self.text1 = None
        self.text2 = None
        self.llm = CTransformers(
            model=model_path,
            model_type="llama",
            config={"max_new_tokens": 512, "temperature": 0.1}
        )
        self.chain = self._build_chain()
        print('Chain made!')

    def _build_chain(self):
        prompt = ChatPromptTemplate([
            ("system", "You are a strict AI that compares the sentences of Text from PDF1 and Text from PDF2 line by line. Carefully compare the following two texts. Look for even the smallest differences: words, punctuation, order, or formatting."
            "Are these two texts exactly the same? Reply with - Pass if they are 100 percent identical."
            "- Fail if there is **any** difference"
            "Output format: [Pass or Fail only]"
            "Reply should strictly be only one word either **Pass** or **Fail**. No explanation needed. Do not include extra information words like **Response:** or **Assistant:** or **AI:** in the response."),
            ("user", "are the two given texts PDF1 and PDF2 exactly same? PDF1: {text1} and PDF2: {text2}.")
        ])
        return LLMChain(llm=self.llm, prompt=prompt)

    def extract_text(self, pdf_file: BytesIO) -> str:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

        # Normalize all whitespace to a single space
        clean_text = re.sub(r'\s+', ' ', text).strip()
        return clean_text

    def compare_pdfs(self, pdf_path1: str, pdf_path2: str) -> str:
        print('Inside compare pdfs')
        self.text1 = self.extract_text(pdf_path1)
        self.text2 = self.extract_text(pdf_path2)
        print('Extracted both files')
        result = self.chain.run(text1 = self.text1, text2= self.text2)
        print(result.split())
        return "Fail" if "Fail" in result.split()[:4] else "Pass" if "Pass" in result.split()[:4] else "Unclear" #Hard Enforcing the output
    
class PDFComparatorDoc:
    def __init__(self, model_path: str):
        self.llm = CTransformers(
            model=model_path,
            n_ctx=2048,
            temperature=0.1,
            max_tokens=2048,
            verbose=True
        )

        prompt = ChatPromptTemplate([
            ("system", """
                You are a helpful assistant that finds semantic differences between two PDF documents PDF1 and PDF2.

                Instructions:
                Compare the documents and categorize the differences into three sections:
                1. Missing Content → Present in the original but missing in the updated.
                2. Extra Content → Present in the updated but not in the original.
                3. Modified Content → Present in both, but altered.

                Respond in this JSON format:

                {{
                "missing": ["..."],
                "extra": ["..."],
                "modified": [["original text", "updated text"]]
                }}
                The response should only be in valid Json format. Dont include extra information words like **Response:** or **Assistant:**. 
                Strictly follow the rules.
                """),
            ("user", "Find the semantic differences between two PDF documents PDF1: {original} and PDF2: {updated}")
        ])

        self.chain = LLMChain(llm=self.llm, prompt=prompt)

    def extract_text(self, pdf_path: BytesIO) -> str:
        reader = PdfReader(pdf_path)
        return "\n".join([page.extract_text() or "" for page in reader.pages])

    def get_semantic_differences(self, original_text, updated_text):
        response = self.chain.run({"original": original_text, "updated": updated_text})
        print("LLM Response:", response)

        # Safe parsing; assumes valid JSON format
        import json
        try:
            result = json.loads(response)
            return result.get("missing", []), result.get("extra", []), result.get("modified", [])
        except json.JSONDecodeError:
            print("LLM output was not valid JSON.")
            return [], [], []

    def generate_colored_pdf(self, missing, extra, modified, output_path="semantic_diff_output.pdf"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)

        def add_colored_line(text, rgb):
            pdf.set_text_color(*rgb)
            pdf.multi_cell(0, 10, text)

        if missing:
            add_colored_line("Missing Content (Red):", (255, 0, 0))
            for line in missing:
                add_colored_line(f"- {line}", (255, 0, 0))

        if extra:
            add_colored_line("Extra Content (Yellow):", (255, 165, 0))
            for line in extra:
                add_colored_line(f"+ {line}", (255, 165, 0))

        if modified:
            add_colored_line("Modified Content (Blue):", (0, 0, 255))
            for orig, updated in modified:
                add_colored_line(f"Original: {orig}", (100, 100, 100))
                add_colored_line(f"Updated: {updated}", (0, 0, 255))

        pdf.output(output_path)
        print(f"Annotated PDF saved as: {output_path}")
        return output_path

    def compare_pdfs_semantically(self, pdf1_path, pdf2_path):
        text1 = self.extract_text(pdf1_path)
        text2 = self.extract_text(pdf2_path)

        missing, extra, modified = self.get_semantic_differences(text1, text2)
        return self.generate_colored_pdf(missing, extra, modified)