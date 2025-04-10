from model import PDFComparator, PDFComparatorDoc
import streamlit as st
from langchain.document_loaders import PyPDFLoader


# === Main Streamlit App Class === #
class PDFComparatorApp:
    def __init__(self, model_path):
        #self.processor = PDFProcessor()
        self.comparator = PDFComparator(model_path=model_path)


    def run(self):
        st.title("📄 AI-Powered PDF Comparator")

        file1 = st.file_uploader("Upload First PDF", type=["pdf"])
        file2 = st.file_uploader("Upload Second PDF", type=["pdf"])

        if st.button("🔍 Compare") and file1 and file2:
            with st.spinner("Comparing PDFs..."):

                output = self.comparator.compare_pdfs(file1, file2)
                print(output)

            if output == 'Pass':
                st.success("✅ PDFs Match - PASS")
            elif output == 'Fail':
                st.error("❌ PDFs Mismatch - FAIL")
                comparatordoc = PDFComparatorDoc(model_path=model_path)
                report_path = comparatordoc.compare_pdfs_semantically("data\pdf1.pdf", "data\pdf2.pdf")
                with open(report_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Difference Report",
                        data=f,
                        file_name="pdf_diff_report.pdf",
                        mime="application/pdf"
                    )
            else:
                print('Unclear')

# === Entry Point === #
if __name__ == "__main__":
    # Replace this with your actual model path!
    model_path = "models\mistral-7b-instruct-v0.2.Q2_K.gguf"
    app = PDFComparatorApp(model_path=model_path)
    app.run()