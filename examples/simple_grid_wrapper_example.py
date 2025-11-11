# append processed to each line
def grid_process(source_file, target_file):
    with open(source_file, 'r') as src, open(target_file, 'w') as tgt:
        for line in src:
            processed_line = line.strip() + ' processed\n'
            tgt.write(processed_line)