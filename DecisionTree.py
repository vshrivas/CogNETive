from random import seed
from random import randrange

# can add regression by changing terminal nodes to predict average value of y's
# and by changing the cost function to be root mean square error
class TreeNode:
    def __init__(self, feature, threshold, leftData, rightData, depth):
        self.feature = feature
        self.threshold = threshold
        self.lData = leftData
        self.rData = rightData
        self.left = None
        self.right = None
        self.depth = depth

class DecisionTree:
    def __init__(self, max_depth, min_size):
        # find the best split for the data, then recursively keep finding the best
        # split on remaining features
        # feature and a threshold needs to be known
        self.max_depth = max_depth
        self.min_size = min_size

    def try_split(self, feature, threshold, data):
        left, right = [], []
        # each data point
        for row in data:
            if row[feature] < threshold:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # Calculate the Gini index for a split dataset
    def gini_index(self, groups, classes):
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
        	size = float(len(group))
        	# avoid divide by zero
        	if size == 0:
        		continue
        	score = 0.0
        	# score the group based on the score for each class
        	for class_val in classes:
        		p = [row[-1] for row in group].count(class_val) / size
        		score += p * p
        	# weight the group score by its relative size
        	gini += (1.0 - score) * (size / n_instances)
        return gini

    def gini_val(self, grps, classes):
        total_size = float(sum([len(grp) for grp in grps]))
        class_sizes = {}

        for grp in grps.keys():
            class_sizes[grp] = {}
            for row in grps[grp]:
                # if class not in dict keys
                if row[-1] not in class_sizes[grp].keys():
                    class_sizes[grp][row[-1]] = 1
                else:
                    class_sizes[grp][row[-1]] = class_sizes[grp][row[-1]] + 1

        gini_index = 0
        for grp in grps.keys():
            score = 0
            grp_size = float(len(grps[grp])) # get num datapts in grp
            if grp_size == 0: # check if 0
                continue
            for c in classes: # for each class
                numEx = 0
                if c in class_sizes[grp].keys():
                    numEx = class_sizes[grp][c]
                # get prop of samples in this grp in this class
                prop = numEx/grp_size
                #prop = [row[-1] for row in grps[grp]].count(c)/grp_size
                # add prop*prop to score
                score += prop * prop
            # add weighted score to gini index
            gini_index += (1.0 - score) * (grp_size/total_size)

        # return gini index
        return gini_index

    def find_best_split(self, data, depth):
        classes = list(set(row[-1] for row in data))
        bfeature, bthreshold, bgini_val, bleft, bright = -1, -1, 1000, [], []
        # for all features
        for f in range(0, len(data[0])-1):
            print("considering %d classes" % (len(data[0])-1))
            # for each data point, consider its value for this feature a threshold
            for row in data:
                l, r = self.try_split(f, row[f], data)
                gini_val = self.gini_val({"left":l, "right":r}, classes)
                if gini_val < bgini_val:
                    bfeature, bthreshold, bgini_val, bleft, bright = f, row[f], gini_val, l, r

        return TreeNode(bfeature, bthreshold, bleft, bright, depth)

    def make_terminal_node(self, data):
        outcomes = [row[-1] for row in data]
        val = max(set(outcomes), key=outcomes.count)
        print("setting leaf as", val)
        return val

    # recursive function, returns root node
    def create_tree(self, data, depth):
        # find best split of data
        currNode = self.find_best_split(data, depth)
        print("curr depth: %d" % depth)
        print("feature:", currNode.feature, "threshold:", currNode.threshold)
        # no splitting occurred, both left and right nodes should be terminal
        if (len(currNode.lData) == 0) or (len(currNode.rData) == 0):
            print("done splitting this node")
            currNode.left = currNode.right = self.make_terminal_node(currNode.lData + currNode.rData)
            return currNode

        # max depth reached, make left and right nodes terminal
        if currNode.depth >= self.max_depth:
            print("done splitting this node")
            currNode.left = self.make_terminal_node(currNode.lData)
            currNode.right = self.make_terminal_node(currNode.rData)
            return currNode

        # done splitting
        if len(currNode.lData) <= self.min_size:
            print("done splitting left")
            currNode.left = self.make_terminal_node(currNode.lData)
        # continue splitting
        else:
            print("going left")
            currNode.left = self.create_tree(currNode.lData, depth + 1)

        if len(currNode.rData) <= self.min_size:
            print("done splitting right, depth", depth)
            currNode.right = self.make_terminal_node(currNode.rData)
        # continue splitting
        else:
            print("going right")
            currNode.right = self.create_tree(currNode.rData, depth + 1)

        return currNode

def predict(node, datarow):
    # go left
    if datarow[node.feature] < node.threshold:
        if isinstance(node.left, TreeNode):
            print("predicting left, value is %f, threshold is %f" % (datarow[node.feature], node.threshold))
            return predict(node.left, datarow)
        else:
            print("done predicting at left, value is %f, threshold is %f" % (datarow[node.feature], node.threshold))
            return node.left
    # go right
    else:
        if isinstance(node.right, TreeNode):
            print("predicting right, value is %f, threshold is %f" % (datarow[node.feature], node.threshold))
            return predict(node.right, datarow)
        else:
            print("done predicting at right, value is %f, threshold is %f" % (datarow[node.feature], node.threshold))
            return node.right

def test(tree, testset):
    correct = 0.0
    for row in testset:
        print("predict for", row)
        prediction = predict(tree, row)
        if prediction == row[-1]:
            correct += 1
        #print("prediction", prediction)
        #print("class", row[-1])
    accuracy = correct/len(testset)
    print("accuracy: %f" % accuracy)
    return accuracy

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
    	fold = list()
    	while len(fold) < fold_size:
    		index = randrange(len(dataset_copy))
    		fold.append(dataset_copy.pop(index))
    	dataset_split.append(fold)
    return dataset_split

# load and prepare data
filename = 'data_banknote_authentication.txt'
file = open(filename, 'r')
dataset = file.readlines()
data = []
for line in dataset:
    nums = [float(i) for i in line.split(',')]
    data.append(nums)

seed(1)
dataset_split = cross_validation_split(data, 5)
max_depth = 5
min_size = 10
dTree = DecisionTree(max_depth, min_size)
accuracies = []
for i in range(0, 5):
    testset = dataset_split[i]
    datacpy = dataset_split.copy()
    datacpy.remove(testset)
    trainset = sum(datacpy, [])
    tree = dTree.create_tree(trainset, 0)
#testset = dataset_split[0]
#trainset = sum(dataset_split[1:5], [])
    accuracy = test(tree, testset)
    accuracies.append(accuracy)
print(accuracies)
print("average accuracy:", sum(accuracies)/5.0)
#print(dataX)
#print(dataY)
