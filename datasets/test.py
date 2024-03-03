def merge_files(file1_path, file2_path, output_file_path):
    # Read content of file 1 and file 2
    with open(file1_path, 'r') as file1:
        content1 = set(file1.readlines())
    
    with open(file2_path, 'r') as file2:
        content2 = set(file2.readlines())
    
    # Merge content and remove duplicates
    merged_content = content1.union(content2)
    
    # Write merged content to output file
    with open(output_file_path, 'w') as output_file:
        output_file.writelines(merged_content)

# Example usage
file1_path = 'data 7.json'
file2_path = 'data 8.json'
output_file_path = 'data.json'
merge_files(file1_path, file2_path, output_file_path)
