# AI-Art
*Adam Gross and Josh Palmer*
Our project is titled “Generative Models in Generative Art”. Our goal is to approximate
the essence of artistic creativity and productivity, and generative models resonate
philosophically with our approach. We see a beautiful analogy between the propagation of life
and generative art both aesthetically and computationally (generative models are very efficient
representations/storage of complex outputs, as are seeds/eggs). Thus our goal will be to build a
probability distribution that we then feed values for random variables to produce unique, ideally
non-deterministic, outputs. Our approach begins with the question: “What goes into producing
art?” Our initial hypothesis is that the artist’s influences and training are significant, as are the
artist’s responses to setting in which the art is being produced. This includes emotionally,
socially, intellectually, and physically affecting aspects of the setting, and includes conscious
and subconscious reactions. Thus our goal is to develop a system that has a logical
physical/aesthetic basis but also an element, learned from the data, that is unprecedented and
generates outputs that are unique and creative with respect to some basis.
Our learner will render simple objects in certain artistic styles given the object, style, and
an “intangible” attribute that will help facilitate randomness/spontaneity in the art production. A
generative model implies that we will be constructing joint probability distributions, dynamic
Bayes’ networks, and perhaps a Hidden Markov Model. Thus the class of tasks that we want
our learning system to address is producing “new” visual-based art --- drawing from experience
as well as that seemingly intangible “aha” moment that heralds some new idea/perspective. To
make this feasible for the timeline of our course, (for now) this task class will only include
surrealistic and impressionistic styles. This represents a useful concept for our particular class
of tasks because (we hope) through learning from this specific impressionist and surrealist data,
our learner might be able to approximate genuine human/AI imagination. Our data will be tuples
of the following form: (a famous surrealist or impressionist art piece, a picture of very similar
object(s)). We will also need to construct a vector of random variables representing artistic
spontaneity and creativity. We do not have any concrete details yet for this vector. We will
construct the tuple in the following manner: we will scrape high-quality art databases for
prominent artists’ pieces, use pre-existing image-recognition to identify their objects, and then
scrape the web for simple images of these objects. It will be hard to tell how representative our
training data will be of the “true” distribution, because our goal is to produce truly unique
outputs. However, we can have a separate performance metric for testing the data. One option
is to upload the outputs to social media and measure how many likes it gets. Another is to group
our output piece with a set of other art pieces, some of which have a characteristic we want to
test for, and see if we can classify our piece together with other similar ones (using a
pre-existing machine learning classification algorithm if necessary). Our development of the
performance metric will be very crucial because each potential metric can carry loads of implicit
biases. We do not have clear ideas of how to address this yet and will not be able to clarify until
we test some different metrics.
