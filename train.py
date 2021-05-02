import argparse

from agent import Agent

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default='q', help="Can select q or dq for the algorithm")
    parser.add_argument('--run_name', default='baseline', help="pick a name for the experiment")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = args_parse()
    agent = Agent(args)
    print("Training is starting...")
    agent.train(args.algorithm)
    