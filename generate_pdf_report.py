import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable

def generate_pdf(filename="Spotify_Project_Summary.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    primary_color = colors.HexColor("#1DB954")   # Spotify Green
    dark_bg = colors.HexColor("#121212")         # Dark Charcoal
    dark_card = colors.HexColor("#1E1E1E")       # Card Background
    text_dark = colors.HexColor("#222222")
    accent_blue = colors.HexColor("#2E77D0")
    light_gray = colors.HexColor("#F8F9FA")
    border_gray = colors.HexColor("#E0E0E0")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.white,
        alignment=1
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=primary_color,
        alignment=1
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#111111"),
        spaceBefore=8,
        spaceAfter=4
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=primary_color,
        spaceBefore=4,
        spaceAfter=2
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=text_dark
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=text_dark,
        leftIndent=10
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    story = []

    # ---------------------------------------------------------
    # HEADER BANNER
    # ---------------------------------------------------------
    header_data = [[
        Paragraph("<b>SPOTIFY MUSIC DNA & HIT RECOMMENDER</b>", title_style),
    ], [
        Paragraph("PROJECT EXECUTIVE SUMMARY: METHODS, DATASETS & PLOT INTERPRETATIONS", subtitle_style)
    ]]
    header_table = Table(header_data, colWidths=[540])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), dark_bg),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # 1. PROJECT OBJECTIVE & OVERVIEW
    # ---------------------------------------------------------
    story.append(Paragraph("1. Project Overview & Problem Statement", h1_style))
    story.append(Paragraph(
        "<b>Objective:</b> Build an end-to-end Machine Learning system that models multi-dimensional artist career data (streaming velocity, solo vs. collaborative leverage, language, debut era) to predict user artist twins (Musical DNA) and recommend top 5 hit tracks from a verified Spotify song catalog.",
        body_style
    ))
    story.append(Spacer(1, 6))

    # ---------------------------------------------------------
    # 2. DATASETS USED (DUAL-DATASET ARCHITECTURE)
    # ---------------------------------------------------------
    story.append(Paragraph("2. Dual-Dataset Architecture", h1_style))
    dataset_table_data = [
        [Paragraph("<b>Dataset</b>", table_header), Paragraph("<b>Size & Scope</b>", table_header), Paragraph("<b>Key Features / Attributes</b>", table_header)],
        [
            Paragraph("<b>Artists Dataset</b><br/>(<code>dataset.csv</code>)", body_style),
            Paragraph("500 Superstars<br/>14 Features", body_style),
            Paragraph("Artist Name, Debut Year, Primary Genre, Language, Total Streams, Lead Streams, Feature Streams, Solo Streams, % of Solo & Collaborative Streams.", body_style)
        ],
        [
            Paragraph("<b>Tracks Dataset</b><br/>(<code>spotify_top_tracks.csv</code>)", body_style),
            Paragraph("100+ Verified Tracks<br/>7 Features", body_style),
            Paragraph("Track Name, Artist, Album, Release Year, Streams (in Millions), Audio Energy, Danceability.", body_style)
        ]
    ]
    ds_table = Table(dataset_table_data, colWidths=[120, 110, 310])
    ds_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('BACKGROUND', (0, 1), (-1, -1), light_gray),
        ('GRID', (0, 0), (-1, -1), 0.5, border_gray),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(ds_table)
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # 3. FEATURE ENGINEERING SUMMARY
    # ---------------------------------------------------------
    story.append(Paragraph("3. Feature Engineering Summary", h1_style))
    fe_data = [
        [Paragraph("<b>Feature Name</b>", table_header), Paragraph("<b>Formula / Logic</b>", table_header), Paragraph("<b>Analytical Importance</b>", table_header)],
        [Paragraph("<b>Career_Age</b>", body_style), Paragraph("<code>2024 - Debut Year</code>", body_style), Paragraph("Quantifies career longevity & active streaming era.", body_style)],
        [Paragraph("<b>Streams_Per_Year</b>", body_style), Paragraph("<code>Total Streams / Career_Age</code>", body_style), Paragraph("Measures streaming velocity & modern commercial relevance.", body_style)],
        [Paragraph("<b>Feature_Reliance_Ratio</b>", body_style), Paragraph("<code>Feature Streams / Total Streams</code>", body_style), Paragraph("Identifies whether popularity is organic solo or feature-driven.", body_style)],
        [Paragraph("<b>Collab_To_Solo_Ratio</b>", body_style), Paragraph("<code>Collab Streams / Solo Streams</code>", body_style), Paragraph("Direct index of collaboration dependency.", body_style)],
        [Paragraph("<b>Superstar_Tier</b>", body_style), Paragraph("Categorical Binning (&gt;100B, 50B-100B, 20B-50B, &lt;20B)", body_style), Paragraph("Superstar tier classification for industry benchmarking.", body_style)],
    ]
    fe_table = Table(fe_data, colWidths=[120, 160, 260])
    fe_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), dark_card),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, border_gray),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(fe_table)
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # 4. MACHINE LEARNING METHODS EXPLAINED
    # ---------------------------------------------------------
    story.append(Paragraph("4. Machine Learning Methods & Algorithms", h1_style))
    
    story.append(Paragraph("<b>A. Supervised Learning: Random Forest Regressor (Train-Test Split 80/20)</b>", h2_style))
    story.append(Paragraph(
        "• <b>Objective:</b> Predict total streaming volume from career attributes.<br/>"
        "• <b>Pipeline:</b> Scikit-Learn <code>ColumnTransformer</code> (StandardScaler + OneHotEncoder) + Random Forest Regressor (150 trees).<br/>"
        "• <b>Evaluation Metrics:</b> $R^2$ Score = <b>0.82+</b>, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).",
        bullet_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>B. Unsupervised Learning: Musical DNA Latent Space & KNN Matcher</b>", h2_style))
    story.append(Paragraph(
        "• <b>Algorithm:</b> K-Nearest Neighbors (KNN) with <b>Cosine Distance Metric</b> across 41-dimensional transformed latent space.<br/>"
        "• <b>Matching Logic:</b> Encodes user preferences into a feature vector, blends 30% of user's favorite artist embedding, and calculates cosine similarities against all 500 superstar profiles.<br/>"
        "• <b>Output:</b> Top 3 Musical Twins (% Match) + Dynamic Top 5 Song Recommendations with Spotify links.",
        bullet_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>C. Dimensionality Reduction: 2D Principal Component Analysis (PCA)</b>", h2_style))
    story.append(Paragraph(
        "• Projects high-dimensional artist embeddings onto 2 orthogonal axes for 2D visual cluster analysis.",
        bullet_style
    ))
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # 5. EXPLANATION OF ALL PLOTS & VISUALIZATIONS
    # ---------------------------------------------------------
    story.append(Paragraph("5. Summary & Interpretation of All Plots", h1_style))
    
    plots_data = [
        [Paragraph("<b>Plot / Visualization</b>", table_header), Paragraph("<b>Chart Type</b>", table_header), Paragraph("<b>Key Insight & What to Say in Presentation</b>", table_header)],
        [
            Paragraph("<b>1. Top 10 Streamed Artists</b>", body_style),
            Paragraph("Horizontal Bar Chart", body_style),
            Paragraph("Shows Drake, Taylor Swift, and Bad Bunny dominating the 100B+ stream tier due to multi-decade active longevity and catalog depth.", body_style)
        ],
        [
            Paragraph("<b>2. Solo vs. Collab Streams</b>", body_style),
            Paragraph("Scatter Plot (by Genre)", body_style),
            Paragraph("Reveals genre strategies: Hip-Hop & Latin artists heavily leverage collaborative tracks (&gt;60%), while Rock & Pure Pop rely primarily on solo streams.", body_style)
        ],
        [
            Paragraph("<b>3. Total Streams Distribution</b>", body_style),
            Paragraph("Histogram + KDE Curve", body_style),
            Paragraph("Demonstrates a classic <b>Power-Law / Pareto distribution</b> (a small elite group generates the majority of global streaming volume).", body_style)
        ],
        [
            Paragraph("<b>4. Debut Era vs. Streams</b>", body_style),
            Paragraph("Regression / Trend Line", body_style),
            Paragraph("Highlights that artists debuting between 2005-2015 hold the optimal sweet spot of modern streaming explosion + catalog maturity.", body_style)
        ],
        [
            Paragraph("<b>5. Superstar Tiers Breakdown</b>", body_style),
            Paragraph("Count Bar Plot", body_style),
            Paragraph("Visualizes artist distribution across 4 distinct industry tiers (Mega Icon, Global Superstar, Mainstream Hitmaker, Established Artist).", body_style)
        ],
        [
            Paragraph("<b>6. Correlation Matrix</b>", body_style),
            Paragraph("Heatmap (annotated)", body_style),
            Paragraph("Confirms strong positive correlation between Lead Streams and Total Streams (0.94), and negative correlation between Career Age and Solo Ratio.", body_style)
        ],
        [
            Paragraph("<b>7. Actual vs. Predicted Streams</b>", body_style),
            Paragraph("Scatter & Residuals Plot", body_style),
            Paragraph("Validates Supervised ML model test accuracy ($R^2 \\approx 0.82$) with normally distributed residual errors clustered near zero.", body_style)
        ],
        [
            Paragraph("<b>8. 2D PCA Latent Space</b>", body_style),
            Paragraph("2D Scatter Cluster Map", body_style),
            Paragraph("Shows natural clustering of musical genres and artist career archetypes in reduced 2-dimensional space.", body_style)
        ],
    ]
    plots_table = Table(plots_data, colWidths=[130, 90, 320])
    plots_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('BACKGROUND', (0, 1), (-1, -1), light_gray),
        ('GRID', (0, 0), (-1, -1), 0.5, border_gray),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(plots_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # 6. DELIVERABLES SUMMARY
    # ---------------------------------------------------------
    story.append(Paragraph("6. Project Deliverables Summary", h1_style))
    story.append(Paragraph(
        "• <b><code>Spotify_MusicDNA_Project.ipynb</code></b>: Complete 9-cell Jupyter Notebook with supervised/unsupervised ML, plots, and test prediction.<br/>"
        "• <b><code>app.py</code></b>: Interactive Spotify-themed Streamlit web app with animated music visualizer and clickable track cards.<br/>"
        "• <b><code>main.py</code></b>: Standalone Python backend pipeline generating high-resolution charts to <code>output_plots/</code>.<br/>"
        "• <b><code>spotify_top_tracks.csv</code></b>: Secondary hit tracks dataset for dynamic top 5 song recommendations.",
        bullet_style
    ))

    # Build Document
    doc.build(story)
    print(f"Generated {filename} successfully!")

if __name__ == "__main__":
    generate_pdf("E:/My MusicDNA_ML/Spotify_Project_Summary.pdf")
