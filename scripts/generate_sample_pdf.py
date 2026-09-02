"""Skrypt do generowania realistycznego sprawozdania finansowego SEC Form 10-Q w formacie PDF z tabelami."""

from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_nvidia_10q_pdf(output_path: str = "data/pdf_reports/nvidia_q3_fy25_10q.pdf") -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=16,
        leading=20,
        alignment=1,
        textColor=colors.HexColor("#1A365D"),
    )
    subtitle_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        alignment=1,
        textColor=colors.HexColor("#4A5568"),
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=8,
    )

    story = []

    # Nagłówek 10-Q
    story.append(Paragraph("UNITED STATES SECURITIES AND EXCHANGE COMMISSION", subtitle_style))
    story.append(Paragraph("Washington, D.C. 20549", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("FORM 10-Q — QUARTERLY REPORT", title_style))
    story.append(Paragraph("NVIDIA CORPORATION — Q3 FISCAL YEAR 2025 (Period Ended October 27, 2024)", subtitle_style))
    story.append(Spacer(1, 15))

    # Sekcja 1
    story.append(Paragraph("PART I. FINANCIAL INFORMATION", heading_style))
    story.append(Paragraph("Item 1. Condensed Consolidated Statements of Operations (Unaudited)", heading_style))
    story.append(Paragraph(
        "The following table presents our condensed consolidated operating results for the three and nine months ended "
        "October 27, 2024 and October 29, 2023 (in millions, except per share data):",
        body_style,
    ))

    # Tabela 1: Statements of Operations
    table_data = [
        ["Financial Metric (in $ Millions)", "Three Months Ended\nOct 27, 2024", "Three Months Ended\nOct 29, 2023", "YoY Change (%)"],
        ["Total Revenue", "$ 35,082", "$ 18,120", "+94%"],
        ["Cost of Revenue", "$ 8,913", "$ 4,720", "+89%"],
        ["Gross Profit (GAAP)", "$ 26,169", "$ 13,400", "+95%"],
        ["Gross Margin % (GAAP)", "74.6 %", "74.0 %", "+60 bps"],
        ["Research & Development (R&D)", "$ 3,390", "$ 2,294", "+48%"],
        ["Sales, General & Administrative", "$ 897", "$ 690", "+30%"],
        ["Operating Income", "$ 21,882", "$ 10,416", "+110%"],
        ["Net Income", "$ 19,309", "$ 9,243", "+109%"],
    ]

    t1 = Table(table_data, colWidths=[200, 110, 110, 80])
    t1.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1A202C")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t1)
    story.append(Spacer(1, 15))

    # Sekcja 2: Przychody według platform
    story.append(Paragraph("Item 2. Management's Discussion and Analysis of Financial Condition — Revenue by Market Platform", heading_style))
    story.append(Paragraph(
        "We specialize in markets where our computing platforms accelerate enterprise workflows and generative AI. "
        "The following table summarizes revenue by specialized market platform:",
        body_style,
    ))

    t2_data = [
        ["Market Platform Segment", "Q3 FY2025 Revenue", "Q3 FY2024 Revenue", "Primary Growth Drivers"],
        ["Data Center", "$ 30,771 M", "$ 14,514 M", "Hopper H100/H200 demand, InfiniBand Quantum-2 networks"],
        ["Gaming & AI PC", "$ 3,279 M", "$ 2,856 M", "GeForce RTX 40 Series GPUs, back-to-school shipments"],
        ["Professional Visualization", "$ 486 M", "$ 416 M", "Adoption of NVIDIA Omniverse, generative 3D workflows"],
        ["Automotive & Robotics", "$ 449 M", "$ 261 M", "NVIDIA DRIVE Orin adoption by autonomous vehicle OEMs"],
        ["OEM & Other", "$ 97 M", "$ 73 M", "Entry-level embedded compute platforms"],
    ]

    t2 = Table(t2_data, colWidths=[140, 95, 95, 170])
    t2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1A202C")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t2)
    story.append(Spacer(1, 15))

    # Komentarz zarządu i czynniki ryzyka
    story.append(Paragraph("Supply Chain and Blackwell Architecture Update", heading_style))
    story.append(Paragraph(
        "During the third quarter of fiscal year 2025, demand for our Hopper computing platform remained exceptional. "
        "We delivered 13,000 production samples of Blackwell architecture systems to customers including Microsoft, "
        "Oracle Cloud, and CoreWeave. Production ramps for GB200 NVL72 liquid-cooled racks are scheduled for Q4 FY2025. "
        "Our gross margin is expected to moderate slightly to the low-70s range during the initial Blackwell production ramp, "
        "before recovering to mid-70s levels as manufacturing efficiencies at TSMC mature.",
        body_style,
    ))

    story.append(Paragraph("Risk Factors — Export Controls and Foundry Concentration", heading_style))
    story.append(Paragraph(
        "Our business depends critically on independent foundries, primarily Taiwan Semiconductor Manufacturing Company (TSMC), "
        "for wafer fabrication and advanced packaging (CoWoS). Any geopolitical disruption in the Taiwan Strait would severely impair "
        "our ability to supply GPU accelerators. Furthermore, United States Department of Commerce export controls continue to restrict "
        "shipments of advanced computing chips (including modified A800 and H800 architectures) to the People's Republic of China, "
        "capping China's Data Center revenue contribution at mid-single digit percentages.",
        body_style,
    ))

    doc.build(story)
    return str(Path(output_path).resolve())


if __name__ == "__main__":
    out = generate_nvidia_10q_pdf()
    print(f"Generated sample 10-Q PDF: {out}")
