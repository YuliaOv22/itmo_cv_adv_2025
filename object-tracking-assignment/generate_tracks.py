import subprocess

tracks_amount_values = [5, 10, 20]
random_range_values = [2, 5]
bb_skip_percent_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

for tracks_amount in tracks_amount_values:
        for random_range in random_range_values:
            for bb_skip_percent in bb_skip_percent_values:
                # Construct the command
                command = [
                    'python', 'create_track_w_args.py',
                    '--tracks_amount', str(tracks_amount),
                    '--random_range', str(random_range),
                    '--bb_skip_percent', str(bb_skip_percent)
                ]

                print(f"Running with tracks_amount={tracks_amount}, random_range={random_range}, bb_skip_percent={bb_skip_percent}")
                subprocess.run(command)