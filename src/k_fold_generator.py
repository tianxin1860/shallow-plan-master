import sys

# The code draws inspiration from: http://goo.gl/aSbVNV
def k_fold_generator(x, k, folder):
    subset_size = len(x) / k
    for i in range(k):
        train = x[:i*subset_size] + x[(i+1)*subset_size:]
        test = x[i*subset_size:][:subset_size]

        f = open(str(folder)+'/train'+str(i)+'.txt', 'w')
        for t in train:
            f.write(t+'\n')
        f.close()

        f = open(str(folder)+'/test'+str(i)+'.txt', 'w')
        for t in test:
            f.write(t+'\n')
        f.close()

def main(args):
    fl = args[0]
    folder = args[0].split('/')[0]
    f = open(args[0], 'r')
    plans = f.read().split('\n')
    k_fold_generator(plans, 10, folder)

if __name__ == "__main__":
    main(sys.argv[1:])
