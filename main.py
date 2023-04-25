import parse_args
import utils
from tqdm import tqdm

def main(model, opt):
    utils.set_random_seed(opt['random_seed'])
    
    if not opt['test_mode']:
        for epoch in tqdm(range(opt['total_epochs'])):
            model.train()
    
    model.test()

if __name__ == '__main__':
    model, opt = parse_args.collect_args()
    main(model, opt)