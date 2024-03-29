'''
Sentiment Classification & How To "Frame Problems" for a Neural Network
by Andrew Trask

labels.txt  - positive/negative sentiment labels for the associated reviews in reviews.txt
reviews.txt - a collection of 25 thousand movie reviews

Twitter: @iamtrask
Blog: http://iamtrask.github.io

What You Should Already Know
neural networks, forward and back-propagation
stochastic gradient descent
mean squared error
and train/test splits
Where to Get Help if You Need it
Re-watch previous Udacity Lectures
Leverage the recommended Course Reading Material - Grokking Deep Learning (Check inside your classroom for a discount code)

Shoot me a tweet @iamtrask

Tutorial Outline:
Intro: The Importance of "Framing a Problem" (this lesson)

Curate a Dataset

Developing a "Predictive Theory"

1: Quick Theory Validation
Transforming Text to Numbers

2: Creating the Input/Output Data
Putting it all together in a Neural Network (video only - nothing in notebook)

3: Building our Neural Network
Understanding Neural Noise

4: Making Learning Faster by Reducing Noise
Analyzing Inefficiencies in our Network

5: Making our Network Train and Run Faster
Further Noise Reduction

6: Reducing Noise by Strategically Reducing the Vocabulary
Analysis: What's going on in the weights?

'''

#Curate a Dataset
def pretty_print_review_and_label(i):
	print(labels[i] + "\t:\t" + reviews[i][:80] + "...")

g = open('reviews.txt','r') # What we know!
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()

g = open('labels.txt','r') # What we WANT to know!
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()

print(len(reviews))
#print(reviews[0])
print(labels[0])

#Develop a Predictive Theory
print("labels.txt \t : \t reviews.txt\n")

pretty_print_review_and_label(2137)
pretty_print_review_and_label(12816)
pretty_print_review_and_label(6267)
pretty_print_review_and_label(21934)
pretty_print_review_and_label(5297)
pretty_print_review_and_label(4998)

#1. Quick Theory Validation
'''
There are multiple ways to implement these projects, but in order to get your code closer to what Andrew 
shows in his solutions, we've provided some hints and starter code throughout this notebook.
You'll find the Counter class to be useful in this exercise, as well as the numpy library.
'''

from collections import Counter 
import numpy as np 

# We will create 3 counter to store (+ve , -ve reviews and all the word)
positive_counts = Counter()
negative_counts = Counter()
total_counts    = Counter()

'''
TODO: Examine all the reviews. For each word in a positive review, increase the count for that word in both 
your positive counter and the total words counter; likewise, for each word in a negative review, increase 
the count for that word in both your negative counter and the total words counter.

Note: Throughout these projects, you should use split(' ') to divide a piece of text (such as a review) into 
individual words. If you use split() instead, you'll get slightly different results than what the videos and 
solutions show.
'''

# TODO: Loop over all the words in all the reviews and increment the counts in the appropriate counter objects
for i in range(len(reviews)):
	if(labels[i] == 'POSITIVE'):
		for word in reviews[i].split(" "):
			positive_counts[word] += 1
			total_counts[word] += 1
	else:
		for word in reviews[i].split(" "):
			negative_counts[word] += 1
			total_counts[word] += 1

# Examine the counts of the most common words in positive reviews
positive_counts.most_common()

# Examine the counts of the most common words in negative reviews
negative_counts.most_common()

'''
As you can see, common words like "the" appear very often in both positive and negative reviews. Instead of 
finding the most common words in positive or negative reviews, what you really want are the words found in 
positive reviews more often than in negative reviews, and vice versa. To accomplish this, you'll need to 
calculate the ratios of word usage between positive and negative reviews.

TODO: Check all the words you've seen and calculate the ratio of postive to negative uses and store that ratio 
in pos_neg_ratios.

Hint: the positive-to-negative ratio for a given word can be calculated with 
positive_counts[word] / float(negative_counts[word]+1). Notice the +1 in the denominator – that ensures we 
don't divide by zero for words that are only seen in positive reviews.
'''

# Create Counter object to store positive/negative ratios
pos_neg_ratios = Counter()

# TODO: Calculate the ratios of positive and negative uses of the most common words
# Consider words to be "common" if they've been used at least 100 times
for term, cnt in list(total_counts.most_common()):
	if(cnt > 100):
		pos_neg_ratio = positive_counts[term] / float(negative_counts[term]+1)
		pos_neg_ratios[term] = pos_neg_ratio

# Examine the ratios you've calculated for a few words:
print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))

'''
Looking closely at the values you just calculated, we see the following:

Words that you would expect to see more often in positive reviews – like "amazing" – have a ratio greater 
than 1. The more skewed a word is toward postive, the farther from 1 its positive-to-negative ratio will be.
Words that you would expect to see more often in negative reviews – like "terrible" – have positive values 
that are less than 1. The more skewed a word is toward negative, the closer to zero its positive-to-negative 
ratio will be.

Neutral words, which don't really convey any sentiment because you would expect to see them in all sorts of 
reviews – like "the" – have values very close to 1. A perfectly neutral word – one that was used in exactly 
the same number of positive reviews as negative reviews – would be almost exactly 1. The +1 we suggested you 
add to the denominator slightly biases words toward negative, but it won't matter because it will be a tiny 
bias and later we'll be ignoring words that are too close to neutral anyway.

Ok, the ratios tell us which words are used more often in postive or negative reviews, but the specific values 
we've calculated are a bit difficult to work with. A very positive word like "amazing" has a value above 4, 
whereas a very negative word like "terrible" has a value around 0.18. Those values aren't easy to compare for 
a couple of reasons:

Right now, 1 is considered neutral, but the absolute value of the postive-to-negative ratios of very postive 
words is larger than the absolute value of the ratios for the very negative words. So there is no way to 
directly compare two numbers and see if one word conveys the same magnitude of positive sentiment as another 
word conveys negative sentiment. So we should center all the values around netural so the absolute value fro 
neutral of the postive-to-negative ratio for a word would indicate how much sentiment (positive or negative) 
that word conveys.

When comparing absolute values it's easier to do that around zero than one.
To fix these issues, we'll convert all of our ratios to new values using logarithms.

TODO: Go through all the ratios you calculated and convert them to logarithms. (i.e. use np.log(ratio))

In the end, extremely positive and extremely negative words will have positive-to-negative ratios with similar 
magnitudes but opposite signs.
'''

# TODO: Convert ratios to logs
for word, ratio in pos_neg_ratios.most_common():
	pos_neg_ratios[word] = np.log(ratio)

# Examine the new ratios you've calculated for the same words from before:
print("Pos-to-neg ratio log value")
print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))


'''
If everything worked, now you should see neutral words with values close to zero. In this case, "the" is near 
zero but slightly positive, so it was probably used in more positive reviews than negative reviews. But look 
at "amazing"'s ratio - it's above 1, showing it is clearly a word with positive sentiment. And "terrible" has 
a similar score, but in the opposite direction, so it's below -1. It's now clear that both of these words are 
associated with specific, opposing sentiments.

Now run the following cells to see more ratios.

The first cell displays all the words, ordered by how associated they are with postive reviews. 
(Your notebook will most likely truncate the output so you won't actually see all the words in the list.)

The second cell displays the 30 words most associated with negative reviews by reversing the order of the first 
list and then looking at the first 30 words. (If you want the second cell to display all the words, ordered by 
how associated they are with negative reviews, you could just write reversed(pos_neg_ratios.most_common()).)

You should continue to see values similar to the earlier ones we checked – neutral words will be close to 0, 
words will get more positive as their ratios approach and go above 1, and words will get more negative as their 
ratios approach and go below -1. That's why we decided to use the logs instead of the raw ratios.

'''

# words most frequently seen in a review with a "POSITIVE" label
pos_neg_ratios.most_common()

# words most frequently seen in a review with a "NEGATIVE" label
list(reversed(pos_neg_ratios.most_common()))[0:30]

# Note: Above is the code Andrew uses in his solution video, 
#       so we've included it here to avoid confusion.
#       If you explore the documentation for the Counter class, 
#       you will see you could also find the 30 least common
#       words like this: pos_neg_ratios.most_common()[:-31:-1]

from IPython.display import Image

review = "This was a horrible, terrible movie."
Image(filename='sentiment_network.png')

review = "The movie was excellent"
Image(filename='sentiment_network_pos.png')

# Project 2: Creating the Input/Output Data¶
# TODO: Create a set named vocab that contains every word in the vocabulary.
vocab = set(total_counts.keys())
vocab_size = len(vocab)
print(vocab_size)

# TODO: Create a numpy array called layer_0 and initialize it to all zeros. You will find the zeros 
# function particularly helpful here. Be sure you create layer_0 as a 2-dimensional matrix with 1 row and vocab_size columns.
layer_0 = np.zeros((1,vocab_size))
print(layer_0.shape)

Image(filename='sentiment_network.png')

# layer_0 contains one entry for every word in the vocabulary, as shown in the above image. We need to make 
# sure we know the index of each word, so run the following cell to create a lookup table that stores the index of every word.
# Create a dictionary of words in the vocabulary mapped to index positions 
# (to be used in layer_0)

word2index = {}
for i,word in enumerate(vocab):
	word2index[word] = i
    
# display the map of words to indices
word2index

# TODO: Complete the implementation of update_input_layer. It should count how many times each word is used 
# in the given review, and then store those counts at the appropriate indices inside layer_0.

def update_input_layer(review):
	""" 
	Modify the global layer_0 to represent the vector form of review.
	The element at a given index of layer_0 should represent
	how many times the given word occurs in the review.
	Args:
		review(string) - the string of the review
	Returns:
		None
	"""
	global layer_0
	# clear out previous state by resetting the layer to be all 0s
	layer_0 *= 0

	# TODO: count how many times each word is used in the given review and store the results in layer_0 
    for word in review.split(" "):
        layer_0[0][word2index[word]] += 1

# Run the following cell to test updating the input layer with the first review. The indices assigned may not 
# be the same as in the solution, but hopefully you'll see some non-zero values in layer_0.
update_input_layer(reviews[0])
layer_0

# TODO: Complete the implementation of get_target_for_labels. It should return 0 or 1, depending on whether the given label is NEGATIVE or POSITIVE, respectively.

def get_target_for_label(label):
    """Convert a label to `0` or `1`.
    Args:
        label(string) - Either "POSITIVE" or "NEGATIVE".
    Returns:
        `0` or `1`.
    """
    # TODO: Your code here

#Run the following two cells. They should print out'POSITIVE' and 1, respectively.
labels[0]
get_target_for_label(labels[0])

# Run the following two cells. They should print out 'NEGATIVE' and 0, respectively.
labels[1]
get_target_for_label(labels[1])


### Project 3 ### Build a Neural Network
from Neural_Network import SentimentNetwork
# Run the following cell to create a SentimentNetwork that will train on all but the last 1000 reviews 
# (we're saving those for testing). Here we use a learning rate of 0.1.
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)

# Run the following cell to test the network's performance against the last 1000 reviews (the ones we held 
#out from our training set).

# We have not trained the model yet, so the results should be about 50% as it will just be guessing and there are only two possible values to choose from.

mlp.test(reviews[-1000:],labels[-1000:])

#Run the following cell to actually train the network. During training, it will display the model's accuracy 
# repeatedly as it trains so you can see how well it's doing.

mlp.train(reviews[:-1000],labels[:-1000])

#That most likely didn't train very well. Part of the reason may be because the learning rate is too high. 
# Run the following cell to recreate the network with a smaller learning rate, 0.01, and then train the new network.

mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.01)
mlp.train(reviews[:-1000],labels[:-1000])

#That probably wasn't much different. Run the following cell to recreate the network one more time with an 
# even smaller learning rate, 0.001, and then train the new network.

mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.001)
mlp.train(reviews[:-1000],labels[:-1000])

#With a learning rate of 0.001, the network should finall have started to improve during training. It's still 
# not very good, but it shows that this solution has potential. We will improve it in the next lesson.

# End of Project 3.

# Understanding Neural Noise
# The following cells include includes the code Andrew shows in the next video. We've included it here so you 
# can run the cells along with the video without having to type in everything.

def update_input_layer(review):
    
    global layer_0
    
    # clear out previous state, reset the layer to be all 0s
    layer_0 *= 0
    for word in review.split(" "):
        layer_0[0][word2index[word]] += 1

update_input_layer(reviews[0])

layer_0

review_counter = Counter()

for word in reviews[0].split(" "):
	review_counter[word] += 1

review_counter.most_common()

# Project 4: Reducing Noise in our input Data
# TODO: Attempt to reduce the noise in the input data like Andrew did in the previous video. Specifically, do the following:

# Copy the SentimentNetwork class you created earlier into the following cell.
# Modify update_input_layer so it does not count how many times each word is used, but rather just stores 
# whether or not a word was used.

# TODO: -Copy the SentimentNetwork class from Projet 3 lesson
#       -Modify it to reduce noise, like in the video 

#Run the following cell to recreate the network and train it. Notice we've gone back to the higher learning rate of 0.1.

mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)
mlp.train(reviews[:-1000],labels[:-1000])

# That should have trained much better than the earlier attempts. It's still not wonderful, but it should have 
# improved dramatically. Run the following cell to test your model with 1000 predictions.

mlp.test(reviews[-1000:],labels[-1000:])

# End of Project 4.

'''
Andrew's solution was actually in the previous video, so rewatch that video if you had any problems with that 
project. Then continue on to the next lesson.
Analyzing Inefficiencies in our Network
The following cells include the code Andrew shows in the next video. We've included it here so you can run the 
cells along with the video without having to type in everything.
'''
Image(filename='sentiment_network_sparse.png')
layer_0 = np.zeros(10)
layer_0
layer_0[4] = 1
layer_0[9] = 1
layer_0
weights_0_1 = np.random.randn(10,5)
layer_0.dot(weights_0_1)
indices = [4,9]
layer_1 = np.zeros(5)
for index in indices:
    layer_1 += (1 * weights_0_1[index])

layer_1

Image(filename='sentiment_network_sparse_2.png')

layer_1 = np.zeros(5)

for index in indices:
    layer_1 += (weights_0_1[index])

layer_1

# Project 5: Making our Network More Efficient¶

# TODO: Make the SentimentNetwork class more efficient by eliminating unnecessary multiplications and additions
# that occur during forward and backward propagation. To do that, you can do the following:

# Copy the SentimentNetwork class from the previous project into the following cell.

'''
Remove the update_input_layer function - you will not need it in this version.
Modify init_network:
You no longer need a separate input layer, so remove any mention of self.layer_0
You will be dealing with the old hidden layer more directly, so create self.layer_1, a two-dimensional matrix 
with shape 1 x hidden_nodes, with all values initialized to zero 

Modify train:
Change the name of the input parameter training_reviews to training_reviews_raw. This will help with the next step.
At the beginning of the function, you'll want to preprocess your reviews to convert them to a list of indices 
(from word2index) that are actually used in the review. This is equivalent to what you saw in the video when 
Andrew set specific indices to 1. Your code should create a local list variable named training_reviews that 
should contain a list for each review in training_reviews_raw. Those lists should contain the indices for 
words found in the review.

Remove call to update_input_layer

Use self's layer_1 instead of a local layer_1 object.

In the forward pass, replace the code that updates layer_1 with new logic that only adds the weights for the 
indices used in the review.

When updating weights_0_1, only update the individual weights that were used in the forward pass.

Modify run:
Remove call to update_input_layer
Use self's layer_1 instead of a local layer_1 object.
Much like you did in train, you will need to pre-process the review so you can work with word indices, then 
update layer_1 by adding weights for the indices used in the review.
'''

# TODO: -Copy the SentimentNetwork class from Project 4 lesson
#       -Modify it according to the above instructions 

mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)
mlp.train(reviews[:-1000],labels[:-1000])
That should have trained much better than the earlier attempts. Run the following cell to test your model with 1000 predictions.

mlp.test(reviews[-1000:],labels[-1000:])

# End of Project 5.
Image(filename='sentiment_network_sparse_2.png')

# words most frequently seen in a review with a "POSITIVE" label
pos_neg_ratios.most_common()

# words most frequently seen in a review with a "NEGATIVE" label
list(reversed(pos_neg_ratios.most_common()))[0:30]

from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook

output_notebook()

hist, edges = np.histogram(list(map(lambda x:x[1],pos_neg_ratios.most_common())), density=True, bins=100, normed=True)
​
p = figure(tools="pan,wheel_zoom,reset,save",
           toolbar_location="above",
           title="Word Positive/Negative Affinity Distribution")
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="#555555")
show(p)
frequency_frequency = Counter()
​
for word, cnt in total_counts.most_common():
    frequency_frequency[cnt] += 1
hist, edges = np.histogram(list(map(lambda x:x[1],frequency_frequency.most_common())), density=True, bins=100, normed=True)
​
p = figure(tools="pan,wheel_zoom,reset,save",
           toolbar_location="above",
           title="The frequency distribution of the words in our corpus")
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="#555555")
show(p)
#Project 6: Reducing Noise by Strategically Reducing the Vocabulary

TODO: Improve SentimentNetwork's performance by reducing more noise in the vocabulary. Specifically, do the following:
'''
Copy the SentimentNetwork class from the previous project into the following cell.
Modify pre_process_data:
Add two additional parameters: min_count and polarity_cutoff
Calculate the positive-to-negative ratios of words used in the reviews. (You can use code you've written elsewhere in the notebook, but we are moving it into the class like we did with other helper code earlier.)
Andrew's solution only calculates a postive-to-negative ratio for words that occur at least 50 times. This keeps the network from attributing too much sentiment to rarer words. You can choose to add this to your solution if you would like.
Change so words are only added to the vocabulary if they occur in the vocabulary more than min_count times.
Change so words are only added to the vocabulary if the absolute value of their postive-to-negative ratio is at least polarity_cutoff
Modify __init__:
Add the same two parameters (min_count and polarity_cutoff) and use them when you call pre_process_data
'''
# TODO: -Copy the SentimentNetwork class from Project 5 lesson
#       -Modify it according to the above instructions 

# Run the following cell to train your network with a small polarity cutoff.

mlp = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=20,polarity_cutoff=0.05,learning_rate=0.01)
mlp.train(reviews[:-1000],labels[:-1000])

# And run the following cell to test it's performance. It should be

mlp.test(reviews[-1000:],labels[-1000:])
# Run the following cell to train your network with a much larger polarity cutoff.

mlp = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=20,polarity_cutoff=0.8,learning_rate=0.01)
mlp.train(reviews[:-1000],labels[:-1000])
# And run the following cell to test it's performance.

mlp.test(reviews[-1000:],labels[-1000:])

# End of Project 6.¶

mlp_full = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=0,polarity_cutoff=0,learning_rate=0.01)
mlp_full.train(reviews[:-1000],labels[:-1000])

Image(filename='sentiment_network_sparse.png')

def get_most_similar_words(focus = "horrible"):
    most_similar = Counter()
​
    for word in mlp_full.word2index.keys():
        most_similar[word] = np.dot(mlp_full.weights_0_1[mlp_full.word2index[word]],mlp_full.weights_0_1[mlp_full.word2index[focus]])
    
    return most_similar.most_common()

get_most_similar_words("excellent")
get_most_similar_words("terrible")

import matplotlib.colors as colors
​
words_to_visualize = list()
for word, ratio in pos_neg_ratios.most_common(500):
    if(word in mlp_full.word2index.keys()):
        words_to_visualize.append(word)
    
for word, ratio in list(reversed(pos_neg_ratios.most_common()))[0:500]:
    if(word in mlp_full.word2index.keys()):
        words_to_visualize.append(word)
pos = 0
neg = 0
​
colors_list = list()
vectors_list = list()
for word in words_to_visualize:
    if word in pos_neg_ratios.keys():
        vectors_list.append(mlp_full.weights_0_1[mlp_full.word2index[word]])
        if(pos_neg_ratios[word] > 0):
            pos+=1
            colors_list.append("#00ff00")
        else:
            neg+=1
            colors_list.append("#000000")

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0)
words_top_ted_tsne = tsne.fit_transform(vectors_list)

p = figure(tools="pan,wheel_zoom,reset,save",
           toolbar_location="above",
           title="vector T-SNE for most polarized words")
​
source = ColumnDataSource(data=dict(x1=words_top_ted_tsne[:,0],
                                    x2=words_top_ted_tsne[:,1],
                                    names=words_to_visualize,
                                    color=colors_list))
​
p.scatter(x="x1", y="x2", size=8, source=source, fill_color="color")
​
word_labels = LabelSet(x="x1", y="x2", text="names", y_offset=6,
                  text_font_size="8pt", text_color="#555555",
                  source=source, text_align='center')
p.add_layout(word_labels)
​
show(p)
​
# green indicates positive words, black indicates negative words
