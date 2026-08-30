title: How to think about correlation?  It’s the slope of the regression when x and y have been standardized. | Statistical Modeling, Causal Inference, and Social Science

# [Statistical Modeling, Causal Inference, and Social Science](https://statmodeling.stat.columbia.edu/) {#site-title}

##  {#site-description}

# How to think about correlation? It’s the slope of the regression when x and y have been standardized.

Dave Balan writes:

> I am an economist at the Federal Trade Commission with a very basic statistics question, one that I have put to several fairly high-powered econometricians, and to which no one has had a satisfying answer.
> The question is this. Why are correlations meaningful? We know that they are ubiquitous, they get reported all the time in work across many disciplines. But for the life of me I cannot understand what the question is to which a correlation is the answer. I get that it’s sometimes useful to know whether or not the correlation is close to 0; if it is close to 0 then you know that it’s not too far from the truth to say that no (linear) relationship exists, and that might be all you need to know. By the same token, a correlation of, say, 0.9 tells you that it’s nowhere close to being true that no linear relationship exists, so you need to go further and investigate what that relationship is. What I can’t understand is why people interpret that 0.9 as a meaningful standalone number in its own right. A correlation of 0.9 means that the data lines up pretty nicely along *some* line with a positive slope, *but that slope can be anywhere from just above 0 to just below infinity*. What good does it do to know that a strong linear relationship exists when you have no idea what that relationship is?
> To take the example of your recent (very interesting) election work, a finding that the correlation in the polling errors between State A and State B is 0 would clearly be important and relevant. And so a finding that the correlation is far from 0 is clearly important insofar as it tells you that it’s definitely not OK to assume that it’s zero. But what is its importance beyond that? What good does it do to know that the polling errors between State A and State B are highly correlated if you don’t know whether a 1 percentage point error in state A is associated with an error of 1 percentage point, or 0.1 points, or 2 points in State B?
> I know that correlations have the advantage of being unit-free. And that’s nice, but it doesn’t seem to solve the problem.
> Am I missing something fundamental here? If so, I hope you will share what it is. If not, is it a serious problem? Is there some other unit-free number that could be used instead? Maybe something like the elasticities that economists use?

I replied that the way I think about the correlation is that it’s the slope of the regression of y on x if the two variables have been standardized to have the same sd. And I pointed him to section 12.3 of Regression and Other Stories, which discusses this point.

Balan followed up:

> Below is my \[Balan’s\] attempt at some intuition:
> A. Since the correlation is the common slope of the y-on-x regression line and the x-on-y regression line, the dots must be configured in such a way that they look pretty much the same if you flip the axes.
> B. The only way that that can be true is if the dots lie around some line with a slope of 1.
> C. Note that this does NOT mean that the regression line through those dots is 1, rather it has to be and the randomness means that the dots will not line up perfectly along that line.

To which I responded: Yes, corr is like a rescaled regression coefficient. Sometimes this makes sense, other times it does not. For example if you are computing elasticity, which is roughly speaking the regression of log(output) on log(input), then standardization would make no sense at all. But if x and y are two different standardized tests, it could make sense to renorm each to have mean 0 and sd 1.
