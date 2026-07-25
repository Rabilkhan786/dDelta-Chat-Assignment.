from pathlib import Path
import fitz

from src.ingest.pdf_native import NativePDFAdapter


def main():

    pdf_path = Path(r"D:\Delta_Chat\data\input\Lift Gas compressor-P&ID.pdf")

    adapter = NativePDFAdapter()

    if not adapter.supports(pdf_path):
        print("Unsupported file")
        return

    # Parse the document
    document = adapter.parse(pdf_path)

    # Open PDF directly using PyMuPDF
    pdf = fitz.open(pdf_path)

    print("=" * 70)
    print("Verification")
    print("=" * 70)

    for page_index, pdf_page in enumerate(pdf):

        blocks = pdf_page.get_text("blocks")
        elements = document.pages[page_index].elements

        non_empty_blocks = []

        for block in blocks:
            text = block[4]

            if text.strip():
                non_empty_blocks.append(block)

        print(f"\nPage {page_index + 1}")
        print(f"Total PDF Blocks      : {len(blocks)}")
        print(f"Non-empty PDF Blocks  : {len(non_empty_blocks)}")
        print(f"Canonical Elements    : {len(elements)}")

        if len(non_empty_blocks) == len(elements):
            print("PASS: Every non-empty PDF block became one Canonical Element.")
        else:
            print("WARNING: Some non-empty blocks were skipped.")

            element_texts = {e.text.strip() for e in elements}

            print("\nSkipped Blocks:")
            print("-" * 70)

            for block in non_empty_blocks:
                text = block[4].strip()

                if text not in element_texts:
                    print(repr(text))

    pdf.close()


if __name__ == "__main__":
    main()