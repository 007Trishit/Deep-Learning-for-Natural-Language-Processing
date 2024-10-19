import csv
import os
import sys
import glob


def tsv_to_csv(input_file):
    # Generate the output filename by replacing .tsv with .csv
    output_file = input_file.rsplit('.', 1)[0] + '.csv'

    try:
        with open(input_file, 'r', newline='') as tsv_file, \
                open(output_file, 'w', newline='') as csv_file:

            tsv_reader = csv.reader(tsv_file, delimiter='\t')
            csv_writer = csv.writer(csv_file)

            for row in tsv_reader:
                csv_writer.writerow(row)

        print(f"Converted: {input_file} -> {output_file}")
    except IOError as e:
        print(f"Error converting {input_file}: {e}")


def convert_multiple_files(input_pattern):
    tsv_files = glob.glob(input_pattern)

    if not tsv_files:
        print(f"No files found matching the pattern: {input_pattern}")
        return

    for tsv_file in tsv_files:
        tsv_to_csv(tsv_file)

    print(f"Conversion complete. Processed {len(tsv_files)} file(s).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_pattern>")
        print("Example: python script.py '*.tsv' (use quotes to prevent shell expansion)")
        sys.exit(1)

    input_pattern = sys.argv[1]
    convert_multiple_files(input_pattern)
