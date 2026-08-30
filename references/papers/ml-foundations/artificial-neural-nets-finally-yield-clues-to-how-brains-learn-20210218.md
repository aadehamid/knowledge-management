title: Artificial Neural Nets Finally Yield Clues to How Brains Learn
description: The learning algorithm that enables the runaway success of deep neural networks doesn’t work in biological brains, but researchers are finding alternatives that could.

# Artificial Neural Nets Finally Yield Clues to How Brains Learn

*February 18, 2021*

![](https://www.quantamagazine.org/wp-content/uploads/2021/02/Neuron_2880x1620_Lede_Still.jpg)

In 2007, some of the leading thinkers behind deep neural networks organized an unofficial “satellite” meeting at the margins of a prestigious annual conference on artificial intelligence. The conference had rejected their request for an official workshop; deep neural nets were still a few years away from taking over AI. The bootleg meeting’s final speaker was [Geoffrey Hinton](https://www.cs.toronto.edu/~hinton/) of the University of Toronto, the cognitive psychologist and computer scientist responsible for some of the biggest breakthroughs in deep nets. He started with a quip: “So, about a year ago, I came home to dinner, and I said, ‘I think I finally figured out how the brain works,’ and my 15-year-old daughter said, ‘Oh, Daddy, not again.’”

The audience laughed. Hinton continued, “So, here’s how it works.” More laughter ensued.

Hinton’s jokes belied a serious pursuit: using AI to understand the brain. Today, deep nets rule AI in part because of an algorithm called backpropagation, or backprop. The algorithm enables deep nets to learn from data, endowing them with the ability to classify images, recognize speech, translate languages, make sense of road conditions for self-driving cars, and accomplish a host of other tasks.

But real brains are highly unlikely to be relying on the same algorithm. It’s not just that “brains are able to generalize and learn better and faster than the state-of-the-art AI systems,” said [Yoshua Bengio](https://yoshuabengio.org/), a computer scientist at the University of Montreal, the scientific director of Mila, the Quebec Artificial Intelligence Institute and one of the organizers of the 2007 workshop. For a variety of reasons, backpropagation isn’t compatible with the brain’s anatomy and physiology, particularly in the cortex.

Bengio and many others inspired by Hinton have been thinking about more biologically plausible learning mechanisms that might at least match the success of backpropagation. Three of them — feedback alignment, equilibrium propagation and predictive coding — have shown particular promise. Some researchers are also incorporating the properties of certain types of cortical neurons and processes such as attention into their models. All these efforts are bringing us closer to understanding the algorithms that may be at work in the brain.

“The brain is a huge mystery. There’s a general impression that if we can unlock some of its principles, it might be helpful for AI,” said Bengio. “But it also has value in its own right.”

## Learning Through Backpropagation

For decades, neuroscientists’ theories about how brains learn were guided primarily by a rule introduced in 1949 by the Canadian psychologist Donald Hebb, which is often paraphrased as “Neurons that fire together, wire together.” That is, the more correlated the activity of adjacent neurons, the stronger the synaptic connections between them. This principle, with some modifications, was successful at explaining certain limited types of learning and visual classification tasks.

But it worked far less well for large networks of neurons that had to learn from mistakes; there was no directly targeted way for neurons deep within the network to learn about discovered errors, update themselves and make fewer mistakes. “The Hebbian rule is a very narrow, particular and not very sensitive way of using error information,” said [Daniel Yamins](https://web.stanford.edu/~yamins/), a computational neuroscientist and computer scientist at Stanford University.

Nevertheless, it was the best learning rule that neuroscientists had, and even before it dominated neuroscience, it inspired the development of the first artificial neural networks in the late 1950s. Each artificial neuron in these networks receives multiple inputs and produces an output, like its biological counterpart. The neuron multiplies each input with a so-called “synaptic” weight — a number signifying the importance assigned to that input — and then sums up the weighted inputs. This sum is the neuron’s output. By the 1960s, it was clear that such neurons could be organized into a network with an input layer and an output layer, and the artificial neural network could be trained to solve a certain class of simple problems. During training, a neural network settled on the best weights for its neurons to eliminate or minimize errors.

However, it was obvious even in the 1960s that solving more complicated problems required one or more “hidden” layers of neurons sandwiched between the input and output layers. No one knew how to effectively train artificial neural networks with hidden layers — until 1986, when Hinton, the late David Rumelhart and [Ronald Williams](https://www.ccs.neu.edu/home/rjw/) (now of Northeastern University) published [the backpropagation algorithm](https://doi.org/10.1038/323533a0).

The algorithm works in two phases. In the “forward” phase, when the network is given an input, it infers an output, which may be erroneous. The second “backward” phase updates the synaptic weights, bringing the output more in line with a target value.

To understand this process, think of a “loss function” that describes the difference between the inferred and desired outputs as a landscape of hills and valleys. When a network makes an inference with a given set of synaptic weights, it ends up at some location on the loss landscape. To learn, it needs to move down the slope, or gradient, toward some valley, where the loss is minimized to the extent possible. Backpropagation is a method for updating the synaptic weights to descend that gradient.

In essence, the algorithm’s backward phase calculates how much each neuron’s synaptic weights contribute to the error and then updates those weights to improve the network’s performance. This calculation proceeds sequentially backward from the output layer to the input layer, hence the name backpropagation. Do this over and over for sets of inputs and desired outputs, and you’ll eventually arrive at an acceptable set of weights for the entire neural network.

## Impossible for the Brain

The invention of backpropagation immediately elicited an outcry from some neuroscientists, who said it could never work in real brains. The most notable naysayer was Francis Crick, the Nobel Prize-winning co-discoverer of the structure of DNA who later became a neuroscientist. In 1989 [Crick wrote](https://doi.org/10.1038/337129a0), “As far as the learning process is concerned, it is unlikely that the brain actually uses back propagation.”

Backprop is considered biologically implausible for several major reasons. The first is that while computers can easily implement the algorithm in two phases, doing so for biological neural networks is not trivial. The second is what computational neuroscientists call the weight transport problem: The backprop algorithm copies or “transports” information about all the synaptic weights involved in an inference and updates those weights for more accuracy. But in a biological network, neurons see only the outputs of other neurons, not the synaptic weights or internal processes that shape that output. From a neuron’s point of view, “it’s OK to know your own synaptic weights,” said Yamins. “What’s not okay is for you to know some other neuron’s set of synaptic weights.”

Any biologically plausible learning rule also needs to abide by the limitation that neurons can access information only from neighboring neurons; backprop may require information from more remote neurons. So “if you take backprop to the letter, it seems impossible for brains to compute,” said Bengio.

Nonetheless, Hinton and a few others immediately took up the challenge of working on biologically plausible variations of backpropagation. “The first paper arguing that brains do \[something like\] backpropagation is about as old as backpropagation,” said [Konrad Kording](https://psychology.sas.upenn.edu/people/konrad-kording), a computational neuroscientist at the University of Pennsylvania. Over the past decade or so, as the successes of artificial neural networks have led them to dominate artificial intelligence research, the efforts to find a biological equivalent for backprop have intensified.

## Staying More Lifelike

Take, for example, one of the strangest solutions to the weight transport problem, courtesy of [Timothy Lillicrap](http://contrastiveconvergence.net/~timothylillicrap/index.php) of Google DeepMind in London and his colleagues in 2016. Their algorithm, instead of relying on a matrix of weights recorded from the forward pass, used a matrix initialized with random values for the backward pass. Once assigned, these values never change, so no weights need to be transported for each backward pass.

To almost everyone’s surprise, the network learned. Because the forward weights used for inference are updated with each backward pass, the network still descends the gradient of the loss function, but by a different path. The forward weights slowly align themselves with the randomly selected backward weights to eventually yield the correct answers, giving the algorithm its name: feedback alignment.

“It turns out that, actually, that doesn’t work as bad as you might think it does,” said Yamins — at least for simple problems. For large-scale problems and for deeper networks with more hidden layers, feedback alignment doesn’t do as well as backprop: Because the updates to the forward weights are less accurate on each pass than they would be from truly backpropagated information, it takes much more data to train the network.

Researchers have also explored ways of matching the performance of backprop while maintaining the classic Hebbian learning requirement that neurons respond only to their local neighbors. Backprop can be thought of as one set of neurons doing the inference and another set of neurons doing the computations for updating the synaptic weights. Hinton’s idea was to work on algorithms in which each neuron was doing both sets of computations. “That was basically what Geoff’s talk was \[about\] in 2007,” said Bengio.

Building on Hinton’s work, Bengio’s team proposed a learning rule in 2017 that requires a neural network with recurrent connections (that is, if neuron A activates neuron B, then neuron B in turn activates neuron A). If such a network is given some input, it sets the network reverberating, as each neuron responds to the push and pull of its immediate neighbors.

Eventually, the network reaches a state in which the neurons are in equilibrium with the input and each other, and it produces an output, which can be erroneous. The algorithm then nudges the output neurons toward the desired result. This sets another signal propagating backward through the network, setting off similar dynamics. The network finds a new equilibrium.

“The beauty of the math is that if you compare these two configurations, before the nudging and after nudging, you’ve got all the information you need to find the gradient,” said Bengio. Training the network involves simply repeating this process of “equilibrium propagation” iteratively over lots of labeled data.

## Predicting Perceptions

The constraint that neurons can learn only by reacting to their local environment also finds expression in new theories of how the brain perceives. [Beren Millidge](https://www.inf.ed.ac.uk/people/students/Beren_Millidge.html), a doctoral student at the University of Edinburgh and a visiting fellow at the University of Sussex, and his colleagues have been reconciling this new view of perception — called predictive coding — with the requirements of backpropagation. “Predictive coding, if it’s set up in a certain way, will give you a biologically plausible learning rule,” said Millidge.

Predictive coding posits that the brain is constantly making predictions about the causes of sensory inputs. The process involves hierarchical layers of neural processing. To produce a certain output, each layer has to predict the neural activity of the layer below. If the highest layer expects to see a face, it predicts the activity of the layer below that can justify this perception. The layer below makes similar predictions about what to expect from the one beneath it, and so on. The lowest layer makes predictions about actual sensory input — say, the photons falling on the retina. In this way, predictions flow from the higher layers down to the lower layers.

But errors can occur at each level of the hierarchy: differences between the prediction that a layer makes about the input it expects and the actual input. The bottommost layer adjusts its synaptic weights to minimize its error, based on the sensory information it receives. This adjustment results in an error between the newly updated lowest layer and the one above, so the higher layer has to readjust its synaptic weights to minimize its prediction error. These error signals ripple upward. The network goes back and forth, until each layer has minimized its prediction error.

Millidge has shown that, with the proper setup, predictive coding networks can converge on much the same learning gradients as backprop. “You can get really, really, really close to the backprop gradients,” he said.

However, for every backward pass that a traditional backprop algorithm makes in a deep neural network, a predictive coding network has to iterate multiple times. Whether or not this is biologically plausible depends on exactly how long this might take in a real brain. Crucially, the network has to converge on a solution before the inputs from the world outside change.

“It can’t be like, ‘I’ve got a tiger leaping at me, let me do 100 iterations back and forth, up and down my brain,’” said Millidge. Still, if some inaccuracy is acceptable, predictive coding can arrive at generally useful answers quickly, he said.

## Pyramidal Neurons

Some scientists have taken on the nitty-gritty task of building backprop-like models based on the known properties of individual neurons. Standard neurons have dendrites that collect information from the axons of other neurons. The dendrites transmit signals to the neuron’s cell body, where the signals are integrated. That may or may not result in a spike, or action potential, going out on the neuron’s axon to the dendrites of post-synaptic neurons.

But not all neurons have exactly this structure. In particular, pyramidal neurons — the most abundant type of neuron in the cortex — are distinctly different. Pyramidal neurons have a treelike structure with two distinct sets of dendrites. The trunk reaches up and branches into what are called apical dendrites. The root reaches down and branches into basal dendrites.

Models developed independently by Kording in 2001, and more recently by [Blake Richards](https://www.mcgill.ca/neuro/blake-richards-phd) of McGill University and Mila and his colleagues, have shown that pyramidal neurons could form the basic units of a deep learning network by doing both forward and backward computations simultaneously. The key is in the separation of the signals entering the neuron for forward-going inference and for backward-flowing errors, which could be handled in the model by the basal and apical dendrites, respectively. Information for both signals can be encoded in the spikes of electrical activity that the neuron sends down its axon as an output.

In the latest work from Richards’ team, “we’ve gotten to the point where we can show that, using fairly realistic simulations of neurons, you can train networks of pyramidal neurons to do various tasks,” said Richards. “And then using slightly more abstract versions of these models, we can get networks of pyramidal neurons to learn the sort of difficult tasks that people do in machine learning.”

## The Role of Attention

An implicit requirement for a deep net that uses backprop is the presence of a “teacher”: something that can calculate the error made by a network of neurons. But “there is no teacher in the brain that tells every neuron in the motor cortex, ‘You should be switched on and you should be switched off,’” said [Pieter Roelfsema](https://nin.nl/research/researchgroups/roelfsema-group/) of the Netherlands Institute for Neuroscience in Amsterdam.

Roelfsema thinks the brain’s solution to the problem is in the process of attention. In the late 1990s, he and his colleagues showed that when monkeys fix their gaze on an object, neurons that represent that object in the cortex become more active. The monkey’s act of focusing its attention produces a feedback signal for the responsible neurons. “It is a highly selective feedback signal,” said Roelfsema. “It’s not an error signal. It is just saying to all those neurons: You’re going to be held responsible \[for an action\].”

Roelfsema’s insight was that this feedback signal could enable backprop-like learning when combined with processes revealed in certain other neuroscientific findings. For example, [Wolfram Schultz](https://www.neuroscience.cam.ac.uk/directory/profile.php?Schultz) of the University of Cambridge and others have shown that when animals perform an action that yields better results than expected, the brain’s dopamine system is activated. “It floods the whole brain with neural modulators,” said Roelfsema. The dopamine levels act like a global reinforcement signal.

In theory, the attentional feedback signal could prime only those neurons responsible for an action to respond to the global reinforcement signal by updating their synaptic weights, said Roelfsema. He and his colleagues have used this idea to build a deep neural network and study its mathematical properties. “It turns out you get error backpropagation. You get basically the same equation,” he said. “But now it became biologically plausible.”

The team presented this work at the Neural Information Processing Systems online conference in December. “We can train deep networks,” said Roelfsema. “It’s only a factor of two to three slower than backpropagation.” As such, he said, “it beats all the other algorithms that have been proposed to be biologically plausible.”

Nevertheless, concrete empirical evidence that living brains use these plausible mechanisms remains elusive. “I think we’re still missing something,” said Bengio. “In my experience, it could be a little thing, maybe a few twists to one of the existing methods, that’s going to really make a difference.”

Meanwhile, Yamins and his colleagues at Stanford have suggestions for how to determine which, if any, of the proposed learning rules is the correct one. By analyzing 1,056 artificial neural networks implementing different models of learning, they found that the type of learning rule governing a network can be identified from the activity of a subset of neurons over time. It’s possible that such information could be recorded from monkey brains. “It turns out that if you have the right collection of observables, it might be possible to come up with a fairly simple scheme that would allow you to identify learning rules,” said Yamins.

Given such advances, computational neuroscientists are quietly optimistic. “There are a lot of different ways the brain could be doing backpropagation,” said Kording. “And evolution is pretty damn awesome. Backpropagation is useful. I presume that evolution kind of gets us there.”

 *Get highlights of the most important news delivered to your email inbox* 

## Comment on this article

## Next article
