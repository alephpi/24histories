import os
import pprint
import shutil
import zipfile
from typing import List

from lxml import etree

import docx
from docx.document import Document
from docx.section import Section
from docx.table import Table
from docx.text.paragraph import Paragraph

# remove blank lines as following will break section structure
# def remove_blank_lines_in_section(doc: Document, section_index: int):
#     """
#     Removes all blank lines within the specified section of the Word document.
#     """
#     section = doc.sections[section_index]

#     # Iterate over paragraphs in the section
#     for paragraph in section.iter_inner_content():
#     # for paragraph in list(section.iter_inner_content())[:-1]:
#         if not paragraph.text.strip():
#             # Remove the blank paragraph
#             # paragraph._element.getparent().remove(paragraph._element)
#             paragraph._element = None

# def remove_empty_paragraphs(doc: Document):
#     """
#     Remove all empty paragraphs from the given Word document.
#     """
#     for paragraph in doc.paragraphs[::-1]:  # Iterate over paragraphs in reverse order
#         if not paragraph.text.strip():  # Check if the paragraph is empty
#             paragraph._element.getparent().remove(
#                 paragraph._element
#             )  # Remove the paragraph element

def preprocessing(file_name: str):
    # Extract the DOCX file contents
    with zipfile.ZipFile(file_name, 'r') as zip:
        zip.extractall('temp')

    # Modify the document.xml file
    doc_xml = os.path.join('temp', 'word', 'document.xml')
    with open(doc_xml, 'r', encoding='utf-8') as f:
        doc = f.read()
    # return doc
    doc_cleaned = clean_doc(doc)
    with open(doc_xml, 'w', encoding='utf-8') as f:
        f.write(doc_cleaned)
    # Recompress the DOCX file
    with zipfile.ZipFile(file_name, 'w') as new_zip:
        for root, dirs, files in os.walk('temp'):
            for file in files:
                path = os.path.join(root, file)
                new_zip.write(path, os.path.relpath(path, 'temp'))

    # remove temp folder
    shutil.rmtree('temp')


def clean_doc(doc: str):
    # Register namespaces
    ns = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
        # Add other namespaces as needed
    }

    # Parse the XML data
    root = etree.fromstring(doc.encode('utf-8'))

    # Find all paragraphs that contain mc:AlternateContent elements
    false_paragraphs = root.xpath(
        ".//w:p[.//mc:AlternateContent]",
        namespaces=ns,
    )

    for false_paragraph in false_paragraphs:
        true_paragraphs = false_paragraph.xpath(
            ".//mc:Fallback//w:p",
            namespaces=ns,
        )
        # get the parent of false_paragraph, remove the false_paragraph and insert all the true_paragraphs
        parent = false_paragraph.getparent()
        index = parent.index(false_paragraph)
        parent.remove(false_paragraph)
        for true_paragraph in true_paragraphs:
            parent.insert(index, true_paragraph)
            index += 1

    # Output the modified XML with original namespace prefixes
    doc_cleaned = etree.tostring(root, xml_declaration=True, encoding="utf-8").decode("utf-8")
    return doc_cleaned

def show_doc(doc: Document, sec_range=(None, None), show_empty=False):
    for i, sec in enumerate(doc.sections[sec_range[0]:sec_range[1]]):
        print(f"----- Section {i} -----")
        show_sec(sec, show_empty)

def show_sec(sec: Section, show_empty=False):
    for i, para in enumerate(sec.iter_inner_content()):
        show_para(para, show_empty)

def show_para(para: Paragraph|Table|List[Paragraph|Table], show_empty=False):
    if isinstance(para, Table):
        print(para.table)
    elif isinstance(para, Paragraph):
        left_indent, first_line_indent, font_size = para_property(para)
        if not show_empty:
            if para.text.strip():
                print(left_indent, first_line_indent, font_size, para.text)
        else:
            print(left_indent, first_line_indent, font_size, para.text)
    else:
        for p in para:
            show_para(p, show_empty)


def pagination(doc: Document):
    pages: List[List[Paragraph]] = []
    page: List[Paragraph] = []
    for i, para in enumerate(doc.paragraphs):
        # if para.text contains numbers, it's a section title
        if any(char.isdigit() for char in para.text):
            pages.append(page)
            page = []
        page.append(para)
    return pages

def segmentation(pages: List[List[Paragraph]]):
    new_pages: List[List[List[Paragraph]]] = []
    new_page: List[List[Paragraph]] = []
    section: List[Paragraph] = []
    for page in pages:
        new_page = []
        section = []
        for para in page:
            if para.text.strip():
                section.append(para)
            elif section:
                    new_page.append(section)
                    section = []
        if section:
            new_page.append(section)
        new_pages.append(new_page)
    return new_pages



def para_property(para: Paragraph|Table):
    left_indent = para.paragraph_format.left_indent
    first_line_indent = para.paragraph_format.first_line_indent
    left_indent = left_indent.pt if left_indent else 0
    first_line_indent = first_line_indent.pt if first_line_indent else 0
    font_size = para.runs[0].font.size.pt if para.runs and para.runs[0].font.size else 0
    return left_indent, first_line_indent, font_size
