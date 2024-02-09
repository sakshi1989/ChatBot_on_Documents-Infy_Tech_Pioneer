from pypdf import PdfReader, PdfWriter

def extract_pages(source_pdf_file_path, pages, out_folder):
    pdf_reader = PdfReader(source_pdf_file_path)
    for page_indices in pages:
        pdf_writer = PdfWriter()  # we want to reset this when starting a new pdf
        for idx in range(page_indices[0] - 1, page_indices[1]):
            pdf_writer.add_page(pdf_reader.pages[idx])
        output_filename = f"{out_folder}/2022-annual-report-wf-trimmed.pdf"
        with open(output_filename, "wb") as out:
            pdf_writer.write(out)

extract_pages("files/2022-annual-report-wf.pdf", [(1, 12)], 'files')