import os
import subprocess

# Define base paths
base_dir = "runs"
dir_base_names = ["FIRST_STAGE", "ZERO_SHOT", "NEUREQ", "SIMPLE_BERT", "CT_MLM_BERT"]
input_dirs = [os.path.join(base_dir, d) for d in dir_base_names]

output_dirs = [f"results/{os.path.basename(d)}" for d in input_dirs]

# Create output directories if they don't exist
for output_dir in output_dirs:
    os.makedirs(output_dir, exist_ok=True)

# Static qrels file
qrels_file = "data/2022/ct_2022_qrels_mapped.txt"

# Run trec_eval for each file in each input_dir
for input_dir, output_dir in zip(input_dirs, output_dirs):
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".txt"):
            retrieval_file = os.path.join(input_dir, file_name)
            output_file = os.path.join(output_dir, file_name)

            # Define trec_eval command with selected metrics
            cmd = [
                "trec_eval",
                "-m", "map",
                "-m", "map_cut",
                "-m", "P",
                "-m", "recall",
                "-m", "ndcg_cut",
                "-m", "recip_rank",
                qrels_file,
                retrieval_file
            ]

            # Run command and save output
            with open(output_file, "w") as out_f:
                subprocess.run(cmd, stdout=out_f, stderr=subprocess.STDOUT)

print("trec_eval run completed with selected metrics.")
