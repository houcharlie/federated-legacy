bazel --output_user_root=/tmp/charliehou-bazel run experiments/emnist:train -- --filtering None --invert_imagery_probability 0p0 --num_client_disc_train_steps $1 --num_server_gen_train_steps $2 --num_clients_per_round $3 --status $4 --num_rounds $5 --client_batch_size $6 --model $7 --optimizer $8 --lr $9 --lr_factor ${10}
