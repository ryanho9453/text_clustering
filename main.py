from optparse import OptionParser
import json
import os
import sys

from flows import Flows

package_path = '/Users/ryanho/Documents/python/focal/text_clustering/find_keyword/'
sys.path.append(package_path)


script_path = os.path.dirname(__file__)


def main():
    with open(os.path.join(script_path, 'config.json'), 'r', encoding='utf8') as f:
        config = json.load(f)

    options = parse_args()
    flows = Flows(config)

    if options.prepare_data:
        flows.prepare_data()

    if options.prepare_build_and_evaluate:
        flows.prepare_build_and_evaluate()

    if options.build_and_evaluate:
        flows.build_and_evaluate()

    if options.tune_alpha_beta:
        flows.tune_alpha_beta()


def parse_args():
    parser = OptionParser(usage='%prog [options]')
    parser.add_option('-p', '--prepare_data', action='store_true', dest='prepare_data',
                      help='prepare_data', default=False)
    parser.add_option('-a', '--prepare_build_and_evaluate', action='store_true', dest='prepare_build_and_evaluate',
                      help='prepare_build_and_evaluate', default=False)
    parser.add_option('-b', '--build_and_evaluate', action='store_true', dest='build_and_evaluate',
                      help='build_and_evaluate', default=False)
    parser.add_option('-t', '--tune_alpha_beta', action='store_true', dest='tune_alpha_beta',
                      help='tune_alpha_beta', default=False)

    options, args = parser.parse_args()

    return options


if __name__ == '__main__':
    main()
