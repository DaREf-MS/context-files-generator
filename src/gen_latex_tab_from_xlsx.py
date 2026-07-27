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
    
    # Safely load the CSV using pandas
    df = pd.read_excel(file_path)
    
    # Updated required columns based on target layout
    required_cols = ['final_category', 'final_sub_category', 'definition']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: '{col}' column not found in the CSV file.")
            return

    total_rows = len(df)
    grouped = df.groupby(['final_category', 'final_sub_category'])
    
    summary_data = []
    for (cat, subcat), group in grouped:
        count = len(group)
        pct = (count / total_rows) * 100
        
        sample_row = group.sample(1).iloc[0]
        
        # Mapping the definition to the Description column
        description = escape_latex(sample_row['definition'])
        
        summary_data.append({
            'final_category': escape_latex(cat),
            'final_sub_category': escape_latex(subcat),
            'Description': description,
            'Count': count,
            'Pct': pct
        })
        
    summary_df = pd.DataFrame(summary_data)
    
    # Calculate total counts per category to sort the table properly
    summary_df['Category_Total'] = summary_df.groupby('final_category')['Count'].transform('sum')
    
    # Sort by Category frequency, then Sub-category frequency, pinning "Others" to the bottom
    is_others = summary_df['final_category'].str.lower() == 'others'
    summary_df = summary_df.assign(Is_Others=is_others).sort_values(
        by=['Is_Others', 'Category_Total', 'Count'], 
        ascending=[True, False, False]
    )
    
    # Build the longtable structure matching the target format exactly
    latex_lines = [
        "\\afterpage{",
        "\\footnotesize",
        "\\begin{longtable}{|p{2.2cm}|p{2.35cm}|p{5.35cm}|r|r@{}|}",
        "\\caption{An overview of developer edits to ACFs according to the identified topics. The distribution of edits highlights frequent omissions in agent instructions} \\label{tab:categorized_data} \\\\",
        "\\hline",
        "\\textbf{Category} & \\textbf{Sub-category} & \\textbf{Description} & \\textbf{\\#} & \\textbf{\\%} \\\\",
        "\\hline",
        "\\endfirsthead",
        "",
        "\\multicolumn{5}{c}{{\\tablename\\ \\thetable{} -- continued from previous page}} \\\\",
        "\\hline",
        "\\textbf{Category} & \\textbf{Sub-category} & \\textbf{Description} & \\textbf{\\#} & \\textbf{\\%} \\\\",
        "\\hline",
        "\\endhead",
        "",
        "\\hline",
        "\\multicolumn{5}{r}{{\\itshape Continued on next page}} \\\\",
        "% \\footempty % Used to prevent duplicate midrules on final page split boundaries",
        "\\endfoot",
        "",
        "\\hline",
        "\\endlastfoot",
        ""
    ]
    
    unique_cats = summary_df['final_category'].unique()
    for cat in unique_cats:
        cat_df = summary_df[summary_df['final_category'] == cat]
        n_rows = len(cat_df)
        
        for i, (_, row) in enumerate(cat_df.iterrows()):
            subcat = row['final_sub_category']
            description = row['Description']
            count = row['Count']
            pct_str = f"{row['Pct']:.1f}\\%"
            
            # Format empty rows (like "Others" frequently is)
            if cat.lower() == 'others' and not row['Description']:
                description = "---"
            
            # Apply \multirow on the first row of the block, centered as requested
            if i == 0:
                col1 = f"\\multirow{{{n_rows}}}{{=}}{{\\centering {cat}}}"
            else:
                col1 = ""
                
            row_str = f" {col1} & {subcat} & {description} & {count} & {pct_str} \\\\"
            latex_lines.append(row_str)
            
            # Use \hhline inside the category block, \hline between categories (adjusted to 4 dashes)
            if i < n_rows - 1:
                latex_lines.append(" \\hhline{~----}")
            
        latex_lines.append(" \\hline")
        latex_lines.append("")
        
    latex_lines.append("\\end{longtable}")
    latex_lines.append("}")
    
    latex_table_code = "\n".join(latex_lines)
    
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(latex_table_code)
        
    print(f"Success! Safe LaTeX table generated and saved to {output_txt_path}\n")
    print(latex_table_code)

if __name__ == "__main__":
    input_file = "data/annotated_acf_edits_final.xlsx"
    generate_latex_table(input_file)