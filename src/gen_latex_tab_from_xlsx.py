import os
import pandas as pd

def escape_latex(text):
    """Escapes common LaTeX special characters in a string."""
    if pd.isna(text):
        return ""
    text = str(text)
    return (text.replace('\\', '\\textbackslash ')
                .replace('%', '\\%')
                .replace('&', '\\&')
                .replace('_', '\\_')
                .replace('#', '\\#'))

def generate_latex_table(file_path, output_txt_path="latex_table.txt"):
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return
    
    # Safely load the Excel using pandas
    df = pd.read_excel(file_path)
    
    required_cols = ['description', 'category', 'sub-category', 'definition', 'example']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: '{col}' column not found in the CSV file.")
            return

    total_rows = len(df)
    grouped = df.groupby(['category', 'sub-category'])
    
    summary_data = []
    for (cat, subcat), group in grouped:
        count = len(group)
        pct = (count / total_rows) * 100
        
        sample_row = group.sample(1).iloc[0]
        
        definition = escape_latex(sample_row['definition'])
        example_text = escape_latex(sample_row['example'])
        
        summary_data.append({
            'Category': escape_latex(cat),
            'Sub-category': escape_latex(subcat),
            'Definition': definition,
            'Example': example_text,
            'Count': count,
            'Pct': pct
        })
        
    summary_df = pd.DataFrame(summary_data)
    
    # Calculate total counts per category to sort the table properly
    summary_df['Category_Total'] = summary_df.groupby('Category')['Count'].transform('sum')
    
    # Sort by Category frequency, then Sub-category frequency, pinning "Others" to the bottom
    is_others = summary_df['Category'].str.lower() == 'others'
    summary_df = summary_df.assign(Is_Others=is_others).sort_values(
        by=['Is_Others', 'Category_Total', 'Count'], 
        ascending=[True, False, False]
    )
    
    # Build the longtable structure matching the target format
    latex_lines = [
        "\\begin{longtable}{|@{}p{2.2cm}|p{2.15cm}|p{3.5cm}|p{4.2cm} |r|r@{}|}",
        "\\caption{An overview of developer edits to ACFs according to the identified topics. The distribution of edits highlights frequent omissions in agent instructions} \\label{tab:categorized_data} \\\\",
        "\\hline",
        "\\textbf{Category} & \\textbf{Sub-category} & \\textbf{Definition} & \\textbf{Example} & \\textbf{\\#} & \\textbf{\\%} \\\\",
        "\\hline",
        "\\endfirsthead",
        "",
        "\\multicolumn{6}{c}{{\\tablename\\ \\thetable{} -- continued from previous page}} \\\\",
        "\\hline",
        "\\textbf{Category} & \\textbf{Sub-category} & \\textbf{Definition} & \\textbf{Example} & \\textbf{\\#} & \\textbf{\\%} \\\\",
        "\\hline",
        "\\endhead",
        "",
        "\\hline",
        "\\multicolumn{6}{r}{{\\itshape Continued on next page}} \\\\",
        "% \\footempty % Used to prevent duplicate midrules on final page split boundaries",
        "\\endfoot",
        "",
        "\\hline",
        "\\endlastfoot",
        ""
    ]
    
    unique_cats = summary_df['Category'].unique()
    for cat in unique_cats:
        cat_df = summary_df[summary_df['Category'] == cat]
        n_rows = len(cat_df)
        
        for i, (_, row) in enumerate(cat_df.iterrows()):
            subcat = row['Sub-category']
            definition = row['Definition']
            example = f"``{row['Example']}''" if row['Example'] else ""
            count = row['Count']
            pct_str = f"{row['Pct']:.1f}\\%"
            
            # Format empty rows (like "Others" frequently is)
            if cat.lower() == 'others' and not row['Definition']:
                definition = "---"
                example = "---"
            
            # Apply \multirow on the first row of the block
            if i == 0:
                col1 = f"\\multirow{{{n_rows}}}{{=}}{{{cat}}}"
            else:
                col1 = ""
                
            row_str = f" {col1} & {subcat} & {definition} & {example} & {count} & {pct_str} \\\\"
            latex_lines.append(row_str)
            
            # Use \hhline inside the category block, \hline between categories
            if i < n_rows - 1:
                latex_lines.append(" \\hhline{~-----}")
            
        latex_lines.append(" \\hline")
        latex_lines.append("")
        
    latex_lines.append("\\end{longtable}\n")
    
    latex_table_code = "\n".join(latex_lines)
    
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(latex_table_code)
        
    print(f"Success! Safe LaTeX table generated and saved to {output_txt_path}\n")
    print(latex_table_code)

if __name__ == "__main__":
    input_file = "data/claude_annotated_data_refined.xlsx"
    generate_latex_table(input_file)