from argparse import ArgumentParser
import matplotlib.pyplot as plt
import unicodedata

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-f", required=True, type=int,
                        help="1 to DQN 2 to DDQN", choices=[1, 2])

    args = parser.parse_args()

    file_type = args.f

    log_path = ""

    if file_type == 1:
        log_path = "./logs/log_dqn.txt"
    elif file_type == 2:
        log_path = "./logs/log_ddqn.txt"
    else:
        print("Invalid file type, exiting...")
        exit()

    episodes = []
    rewards = []
    averages = []
    losses = []

    with open(log_path, "r") as file:
        for line in file:
            line = unicodedata.normalize("NFKD", line)
            line_split = line.split(" | ")
            episodes.append(int(line_split[0][line_split[0].index(":")+1:]))
            rewards.append(float(line_split[1][line_split[1].index(":")+1:]))
            averages.append(float(line_split[2][line_split[2].index(":")+1:]))
            losses.append(float(line_split[3][line_split[3].index(":")+1:]))

    plt.plot(episodes, rewards)
    plt.title("Episode vs. Reward for " + ("DQN" if file_type == 1 else "DDQN"))
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("./plots/e_vs_r_" + ("dqn" if file_type == 1 else "ddqn") + ".jpg", bbox_inches="tight")
    plt.show()

    plt.plot(episodes, averages)
    plt.title("Episode vs. Average Reward for " +
              ("DQN" if file_type == 1 else "DDQN"))
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.savefig("./plots/e_vs_ar_" + ("dqn" if file_type ==
                                      1 else "ddqn") + ".jpg", bbox_inches="tight")
    plt.show()

    plt.plot(episodes, losses)
    plt.title("Episode vs. Loss for " + ("DQN" if file_type == 1 else "DDQN"))
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.savefig("./plots/e_vs_l_" + ("dqn" if file_type ==
                                      1 else "ddqn") + ".jpg", bbox_inches="tight")
    plt.show()
