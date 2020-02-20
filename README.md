# NOvA-CNN-Project
Deep-learning classifier robustness in neutrino experiments- BSc Final Year Project being undertaken under the supervision of Dr. Chris Backhouse.

The NOvA experiment studies the oscillation of muon neutrinos into electron neutrinos, making precise measurements of the neutrino properties, and perhaps shedding light on the origin of the matter/antimatter asymmetry in our universe.

Recently, new classifiers drawn from computer vision research have been deployed to distinguish between neutrino flavours. These have been very succesful, giving a substantial boost to the experiment's sensitivity.

Because these techniques are very new, and the classifiers can be seen as something of a "black box" there are some concerns about the systematic robustness of the technique.

This project is looking as training a classifier invarient to bias from the domain in which the Monte Carlo events are produced, as they are simulated by two different generators, using Domain-Adversarial Training of Neural Networks (Ganin et al, 2015).

The base Convolutional Neural Network Arhcitecture being used it MobileNetV2 (Sandler et all, 2018), with the Keras API being used in Python to train and test the network.
