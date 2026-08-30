title: Ingesting PDFs and why Gemini 2.0 changes everything | Hacker News

# Ingesting PDFs and why Gemini 2.0 changes everything

[Ingesting PDFs and why Gemini 2.0 changes everything](https://www.sergey.fyi/articles/gemini-flash-2)  ([sergey.fyi](https://news.ycombinator.com/from?site=sergey.fyi))

1303 points by [serjester](https://news.ycombinator.com/user?id=serjester) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42952605)  | [hide](https://news.ycombinator.com/hide?id=42952605&goto=item%3Fid%3D42952605) | [past](https://hn.algolia.com/?query=Ingesting%20PDFs%20and%20why%20Gemini%202.0%20changes%20everything&type=story&dateRange=all&sort=byDate&storyText=false&prefix&page=0) | [favorite](https://news.ycombinator.com/fave?id=42952605&auth=3cc485605bc316feaef8edc5ee1659531ec8ed4e) | [447 comments](https://news.ycombinator.com/item?id=42952605)

[lazypenguin](https://news.ycombinator.com/user?id=lazypenguin) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953665)

I work in fintech and we replaced an OCR vendor with Gemini at work for ingesting some PDFs. After trial and error with different models Gemini won because it was so darn easy to use and it worked with minimal effort. I think one shouldn't underestimate that multi-modal, large context window model in terms of ease-of-use. Ironically this vendor is the best known and most successful vendor for OCR'ing this specific type of PDF but many of our requests failed over to their human-in-the-loop process. Despite it not being their specialization switching to Gemini was a no-brainer after our testing. Processing time went from something like 12 minutes on average to 6s on average, accuracy was like 96% of that of the vendor and price was significantly cheaper. For the 4% inaccuracies a lot of them are things like the text "LLC" handwritten would get OCR'd as "IIC" which I would say is somewhat "fair". We probably could improve our prompt to clean up this data even further. Our prompt is currently very simple: "OCR this PDF into this format as specified by this json schema" and didn't require some fancy "prompt engineering" to contort out a result.

Gemini developer experience was stupidly easy. Easy to add a file "part" to a prompt. Easy to focus on the main problem with weirdly high context window. Multi-modal so it handles a lot of issues for you (PDF image vs. PDF with data), etc. I can recommend it for the use case presented in this blog (ignoring the bounding boxes part)!

[kbyatnal](https://news.ycombinator.com/user?id=kbyatnal) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957551)

This is spot on, any legacy vendor focusing on a specific type of PDF is going to get obliterated by LLMs. The problem with using an off-the-shelf provider like this is, you get stuck with their data schema. With an LLM, you have full control over the schema meaning you can parse and extract much more unique data.

The problem then shifts from "can we extract this data from the PDF" to "how do we teach an LLM to extract the data we need, validate its performance, and deploy it with confidence into prod?"

You could improve your accuracy further by adding some chain-of-thought to your prompt btw. e.g. Make each field in your json schema have a \`reasoning\` field beforehand so the model can CoT how it got to its answer. If you want to take it to the next level, \`citations\` in our experience also improves performance (and when combined with bounding boxes, is powerful for human-in-the-loop tooling).

Disclaimer: I started an LLM doc processing infra company ([https://extend.app/](https://extend.app/))

[TeMPOraL](https://news.ycombinator.com/user?id=TeMPOraL) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960720)

\> *The problem then shifts from "can we extract this data from the PDF" to "how do we teach an LLM to extract the data we need, validate its performance, and deploy it with confidence into prod?"*

A smart vendor will shift into that space - they'll use that LLM themselves, and figure out some combination of finetunes, multiple LLMs, classical methods and *human verification* of random samples, that lets them not only "validate its performance, and deploy it with confidence into prod", but also sell that confidence *with an SLA on top of it*.

[wraptile](https://news.ycombinator.com/user?id=wraptile) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42965369)

That's what we did with our web scraping saas - with Extraction API¹ we shifted web scraped data parsing to support both predefined models for common objects like products, reviews etc. and direct LLM prompts that we further optimize for flexible extraction.

There's definitely space here to help the customer realize their extraction vision because it's still hard to scale this effectively on your own!

1 - [https://scrapfly.io/extraction-api](https://scrapfly.io/extraction-api)

[quantumPilot](https://news.ycombinator.com/user?id=quantumPilot) [on Feb 16, 2025](https://news.ycombinator.com/item?id=43072136)

What's the value for a customer to pay a vendor that is only a wrapper around an LLM when they can leverage LLMs directly? I imagine tools being accessible for certain types of users, but for customers like those described here, you're better off replacing any OCR vendor with your own LLM integration

[sitkack](https://news.ycombinator.com/user?id=sitkack) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42962159)

Software is dead, if it isn't a prompt now, it will be a prompt in 6 months.

Most of what we think software is today, will just be a UI. But UIs are also dead.

[SketchySeaBeast](https://news.ycombinator.com/user?id=SketchySeaBeast) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42963150)

I wonder about these takes. Have you never worked in a complex system in a large org before?

OK, sure, we can parse a PDF reliably now, but now we need to act on that data. We need to store it, make sure it ends up with the right people who need to be notified that the data is available for their review. They then need to make decisions upon that data, possible requiring input from multiple stakeholders.

All that back and forth needs to be recorded and stored, along with the eventual decision and the all supporting documents and that whole bundle needs to be made available across multiple systems, which requires a bunch of ETLs and governance.

An LLM with a prompt doesn't replace all that.

[sitkack](https://news.ycombinator.com/user?id=sitkack) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42966251)

We need to think terms of light cones, not dog and pony take downs of whatever system you are currently running. See where thigns are going.

I have worked in large systems, both in code and people, compilers, massive data processing systems, 10k business units.

[collingreen](https://news.ycombinator.com/user?id=collingreen) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42966352)

I don't know what light cones or dog and pony mean here but I'm interested in your take - would you care to expand a bit on how the future can reshape that very complicated set of steps and humans described in the parent?

[SketchySeaBeast](https://news.ycombinator.com/user?id=SketchySeaBeast) [on Feb 7, 2025](https://news.ycombinator.com/item?id=42967995)

I think collingreen followed-up better than I ever could, so I'm hoping you can respond to them with more details.

[victorbjorklund](https://news.ycombinator.com/user?id=victorbjorklund) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42964153)

Can you prompt a salesforce replacement for an org with 100 000 employees?

[mrbungie](https://news.ycombinator.com/user?id=mrbungie) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42966218)

Yesterday I read an /r/singularity post in awe cus of a screenshot of a lead management platform from OAI in a japan convention supposedly meant a direct threat to SalesForce. Like, yeah sure buddy.

I would say most acceleracionist/AI bulls/etc don't really understand the true essential complexity in software development. LLMs are being seen as a software development silver bullets, and we know what happens with silver bullets.

[sitkack](https://news.ycombinator.com/user?id=sitkack) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42966233)

Come back your comment in 18 months.

[collingreen](https://news.ycombinator.com/user?id=collingreen) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42966381)

I assume this is a slap intended to imply that ai actually IS a silver bullet answer to the parent's described problem and in just 18 months they will look back and realize how wrong they are.

Is that what you mean and, if so, is there anything in particular you've seen that leads you to see these problems being solved well or on the 18 month timeline? That sounds interesting to look at to me and I'd love to know more.

[sitkack](https://news.ycombinator.com/user?id=sitkack) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42967637)

It isn't a silver bullet in that it can just "make software" but it is changing the entire dynamic.

You can't do point sampling to figure out where things are going. We have to look at the slope. People see a paper come out, look at the results and say, "this fails for x, y and z. doesn't work", that is now how scientific research works. This is why two minute papers has the tag line, "hold on to your papers ... two papers down the line ..."

Copy and paste the whole thread into a SOTA model and have meta me explain it.

[ethbr1](https://news.ycombinator.com/user?id=ethbr1) [on Feb 8, 2025](https://news.ycombinator.com/item?id=42980359)

That's not why more experienced people are doubting you.

They're doubting you because the *non-digital* portions of processes change at people/org speed.

Which is to say that changing a core business process is a year political consensus, rearchitecture, and change management effort, because you also have to coordinate all the cascading and interfacing changes.

[sitkack](https://news.ycombinator.com/user?id=sitkack) [on Feb 8, 2025](https://news.ycombinator.com/item?id=42985568)

\> changing a core business process is a year political consensus, rearchitecture, and change management effort

You are thinking within the existing structures, those structures will evaporate. All along the software supply chain, processes will get upended, not just because of how technical assets will be created, but also how organizations themselves are structured and react and in turn how software is created and consumed.

This is as big as the invention of the corporation, the printing press and the industrial revolution.

I am not here to tutor people on this viewpoint or defend it, I offer it and everyone can do with it what they will.

[ethbr1](https://news.ycombinator.com/user?id=ethbr1) [on Feb 8, 2025](https://news.ycombinator.com/item?id=42986090)

Ha. Look back on this comment in a few years.

[cpursley](https://news.ycombinator.com/user?id=cpursley) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42962384)

Software without data moats, vender lock-in, etc sure will. All the low handing fruit saas is going to get totally obliterated by LLM built-software.

[fragmede](https://news.ycombinator.com/user?id=fragmede) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42965226)

If I'm an autobody shop or some other well-served niche, how unhappy with them do I have to be to decide to find a replacement, either a competitor of theirs that used an LLM, or bring it in house and go off and find a developer to LLM-acceleratedly make me a better shopmonkey? And there are the integrations. I don't own a low hanging fruit SaaS company, but it seems very sticky, and since the established company already exists, they can just lower prices to meet their competitors.

B2B is different from B2C, so if one vendor has a handful of clients and they won't switch away, there's no obliterating happening.

What's opened up is even lower hanging fruit, on more trees. A SaaS company charging $3/month for the left-handed underwater basket weaver niche now becomes viable as a lifestyle business. The shovels in this could be supabase/similar, since clients can keep access to their data there even if they change frontends.

[sitkack](https://news.ycombinator.com/user?id=sitkack) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42966227)

Which means that the current vc-software-ecosystem is the walking dead. The front end webdev is now going to do things that previously took a 10 person startup.

[cpursley](https://news.ycombinator.com/user?id=cpursley) [on Feb 7, 2025](https://news.ycombinator.com/item?id=42971785)

Integrations is part of the data moat I mentioned.

[Vrondi](https://news.ycombinator.com/user?id=Vrondi) [on Feb 8, 2025](https://news.ycombinator.com/item?id=42983480)

The only thing that will be different for most is vendor lock-in will be to LLM vendors.

[sitkack](https://news.ycombinator.com/user?id=sitkack) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42964952)

Totally agree.

[Cumpiler69](https://news.ycombinator.com/user?id=Cumpiler69) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961020)

*>A smart vendor will shift into that space - they'll use that LLM themselves*

It's a bit late to start shifting now since it takes time. Ideally they should already have a product on the market.

[TeMPOraL](https://news.ycombinator.com/user?id=TeMPOraL) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961190)

There's still time. The situation in which you can effectively replace your OCR vendor with hitting LLM APIs via a half-assed Python script ChatGPT wrote for you, has existed for maybe few months. People are only beginning to realize LLMs got good enough that this is an option. An OCR vendor that starts working on the shift today, should easily be able to develop, tune, test and productize an LLM-based OCR pipeline *way* before most of their customers realize what's been happening.

But it is a good opportunity for a fast-moving OCR service to steal some customers from their competition. If I were working in this space, I'd be worried about that, and also about the possibility some of the LLM companies realize they could actually break into this market themselves right now, and secure some additional income.

EDIT:

I get the feeling that the main LLM suppliers are purposefully sticking to general-purpose APIs and refraining from competing with anyone on specific services, and that this goes beyond just staying focused. Some of potential applications, like OCR, could turn into money printers if they moved on them now, and they all could use some more cash to offset what they burn on compute. Is it because they're trying to avoid starting an "us vs. them" war until after they made everyone else dependent on them?

[anon84873628](https://news.ycombinator.com/user?id=anon84873628) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42965475)

To the point after your edit, I view it like the cloud shift from IaaS to PaaS / SaaS. Start with a neutral infrastructure platform that attracts lots of service providers. Then take your pick of which ones to replicate with a vertically integrated competitor or manager offering once you are too big for anyone to really complain.

[bayindirh](https://news.ycombinator.com/user?id=bayindirh) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961828)

Never underestimate the power of the second mover. Since the development is happening in the open, someone can quickly cobble up the information and cut directly to the 90% of the work.

Then your secret sauce will be your fine tunes, etc.

Like it or not AI/LLM will be a commodity, and this bubble will burst. Moats are hard to build when you have at least one open source copy of what you *just* did.

[SoftTalker](https://news.ycombinator.com/user?id=SoftTalker) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42964485)

And next year your secret sauce will be worthless because the LLMs are that much better again.

Businesses that are just "today's LLM \+ our bespoke improvements" won't have legs.

[pmarreck](https://news.ycombinator.com/user?id=pmarreck) [on Feb 17, 2025](https://news.ycombinator.com/item?id=43081041)

I have some out-of-print books that I want to convert into nice pdf's/epubs (like, reference-quality)

\1) I don't mind destroying the binding to get the best quality. Any idea how I do so?

\2) I have a multipage double-sided scanner (fujitsu scansnap). would this be sufficient to do the scan portion?

\3) Is there anything that determines the font of the book text and reproduces that somehow? and that deals with things like bold and italic and applies that either as markdown output or what have you?

\4) how do you de-paginate the raw text to reflow into (say) an epub or pdf format that will paginate based on the output device (page size/layout) specification?

[raghavsb](https://news.ycombinator.com/user?id=raghavsb) [on Feb 10, 2025](https://news.ycombinator.com/item?id=42999533)

Great, I landed on the reasoning and citations bit through trial and error and the outputs improved for sure.

[MajorData](https://news.ycombinator.com/user?id=MajorData) [on Feb 9, 2025](https://news.ycombinator.com/item?id=42993825)

\`How did you add bounding boxes, especially if it is variety of files?

[bitdribble](https://news.ycombinator.com/user?id=bitdribble) [on Feb 16, 2025](https://news.ycombinator.com/item?id=43069204)

In my open source tool [http://docrouter.ai](http://docrouter.ai) I run both OCR and LLM/Gemini, using litellm to support multiple LLMs. The user can configure extraction schema & prompts, and use tags to select which prompt/llm combination runs on which uploaded PDF.

LLM extractions are searched in OCR output, and if matched, the bounding box is displayed based on OCR output.

Demo: app.github.ai (just register an account and try) Github: [https://github.com/analytiq-hub/doc-router](https://github.com/analytiq-hub/doc-router)

Reach out to me at andrei@analytiqhub.com for questions. Am looking for feedback and collaborators.

[montecruiseto](https://news.ycombinator.com/user?id=montecruiseto) [on Feb 7, 2025](https://news.ycombinator.com/item?id=42971548)

So why should I still use Extend instead of Gemini?

[panta](https://news.ycombinator.com/user?id=panta) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42964598)

How do you handle the privacy of the scanned documents?

[bitdribble](https://news.ycombinator.com/user?id=bitdribble) [on Feb 16, 2025](https://news.ycombinator.com/item?id=43069253)

With the docrouter.ai, it can be installed on prem. If using the SAAS version, users can collaborate in separate workspaces, modeled on how Databricks supports workspaces. Back end DB is Mongo, which keeps things simple.

One level of privacy is the workspace level separation in Mongo. But, if there is customer interest, other setups are possible. E.g. the way Databricks handles privacy is by actually giving each account its own back end services - and scoping workspaces within an account.

That is a good possible model.

[kbyatnal](https://news.ycombinator.com/user?id=kbyatnal) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42964693)

We work with fortune 500s in sensitive industries (healthcare, fintech, etc). Our policies are:

\- data is never shared between customers

\- data never gets used for training

\- we also configure data retention policies to auto-purge after a time period

[panta](https://news.ycombinator.com/user?id=panta) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42965220)

But how to get these guarantees from the upstream vendors? Or do you run the LLMs on premises?

[Karrot\_Kream](https://news.ycombinator.com/user?id=Karrot_Kream) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42965561)

If you're using LLM APIs there are SLAs from the vendors to make sure your inputs are not used as training data and other guarantees. Generally these endpoints cost more to use (the compliance fee essentially) but they solve the problem.

[makeitdouble](https://news.ycombinator.com/user?id=makeitdouble) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956937)

\> After trial and error with different models

As a mere occasional customer I've been scanning 4 to 5 pages of the same document layout every week in gemini for half a year, and every single week the results were slightly different.

To note the docs are bilingual so it could affect the results, but what stroke me is the lack of consistency, and even with the same model, running it two or three times in a row gives different results.

That's fine for my usage, but that sounds like a nightmare if everytime Google tweaks their model, companies have to reajust their whole process to deal with the discrepancies.

And sticking with the same model for multiple years also sound like a captive situation where you'd have to pay premium for Google to keep it available for your use.

[tomrod](https://news.ycombinator.com/user?id=tomrod) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957385)

Consider turning down the temperature in the configuration? LLMs have a bit of randomness in them.

Gemini 2.0 Flash seems better than 1.5 - [https://deepmind.google/technologies/gemini/flash/](https://deepmind.google/technologies/gemini/flash/)

[mejutoco](https://news.ycombinator.com/user?id=mejutoco) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960435)

\> and every single week the results were slightly different.

This is one of the reasons why open source offline models will always be part of the solution, if not the whole solution.

[rafaelmn](https://news.ycombinator.com/user?id=rafaelmn) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960642)

Inconsistency comes from scaling - if you are optimizing your infra to be cos effective you will arrive at same tradeoffs. Not saying it's not nice to be able to make some of those decisions on your own - but if you're picking LLMs for simplicity - we are years away from running your own being in the same league for most people.

[mejutoco](https://news.ycombinator.com/user?id=mejutoco) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961858)

And if you are not you wont.

You can decide if you change your local setup or not. You cannot decide the same of a service.

There is nothing inevitable about inconsistency in a local setup.

[iandanforth](https://news.ycombinator.com/user?id=iandanforth) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957436)

At temperature zero, if you're using the same API/model, this really should not be the case. None of the big players update their APIs without some name / version change.

[pigscantfly](https://news.ycombinator.com/user?id=pigscantfly) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958471)

This isn't really true unfortunately -- mixture of experts routing seems to suffer from batch non-determinism. No one has stated publicly exactly why this is, but you can easily replicate the behavior yourself or find bug reports / discussion with a bit of searching. The outcome and observed behavior of the major closed-weight LLM APIs is that a temperature of zero no longer corresponds to deterministic greedy sampling.

[brookst](https://news.ycombinator.com/user?id=brookst) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959211)

If temperature is zero, and weights are weights, where is the non-deterministic behavior coming from?

[wodenokoto](https://news.ycombinator.com/user?id=wodenokoto) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960416)

Temperature changes the distribution that is sampled, not if a distribution is sampled.

Temperature changes the softmax equation\[1\], not weather or not you are sampling from the softmax result or choosing the highest probability. IBM's documentation corroborates this, saying you need to set do\_sample to True in order for the temperature to have any effect, e.g., T changes how we sample, not if we sample \[2\].

A similar discussion on openai forum also claim that the RNG might be in a different state from run to run, although I am less sure about that \[3\]

\[1\] [https://pelinbalci.com/2023/10/16/Temperature\_parameter.html](https://pelinbalci.com/2023/10/16/Temperature_parameter.html)

\[2\] [https://www.ibm.com/think/topics/llm-temperature#:\~:text\=The...](https://www.ibm.com/think/topics/llm-temperature#:~:text=The%20LLM%20temperature%20parameter%20modifies,of%20selecting%20less%20probable%20tokens.)

\[3\] [https://community.openai.com/t/clarifications-on-setting-tem...](https://community.openai.com/t/clarifications-on-setting-temperature-0/886447/4)

[zelphirkalt](https://news.ycombinator.com/user?id=zelphirkalt) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960586)

I have dealt with traditional ML models in the past and things like tensorflow non-reproducibility. Managed to make them behave reproducibly. This is a very basic requirement. If we cannot even have that or people who deal with Gemini or similar models do not even know why they don't deliver reproducible results ... This seems very bad. It becomes outright unusable for anyone wanting to do research with reliable result. We already have a reproducibility crisis, because researchers often do not have the required knowledge to properly handle their tooling and would need a knowledgeable engineer to set it up. Only that most engineers don't know either and don't show enough attention to the detail to make reproducible software.

[sidkshatriya](https://news.ycombinator.com/user?id=sidkshatriya) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42962521)

Your response is correct. However, you can choose to *not* sample from the distribution. You can have a rule to always choose the token with the highest probability generated by the softmax layer.

This approach should make the LLM deterministic regardless of the temperature chosen.

P.S. Choosing lower and lower temperatures will make the LLM more deterministic but it will never be totally deterministic because there will always be some probability in other tokens. Also it is not possible to use temperature as exactly 0 due to exp(1/T) blowup. Like I mentioned above, you could avoid fiddling with temperature and just decide to always choose token with highest probability for full determinism.

There are probably other more subtle things that might make the LLM non-deterministic from run to run though. It could be due to some non-deterministism in the GPU/CPU hardware. Floating point is very sensitive to ordering.

TL;DR for as much determinism as possible just choose token with highest probability (i.e. *dont* sample the distribution).

[TeMPOraL](https://news.ycombinator.com/user?id=TeMPOraL) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959979)

Here probably routing would be dominating, but in general, unless I missed all the vendors ditching GPUs and switching to ASICs optimized for fixed precision math, floating points are still non-commutative therefore results are non-deterministic wrt. randomness introduced by parallelising the calculations.

[zelphirkalt](https://news.ycombinator.com/user?id=zelphirkalt) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960610)

Of course which part of the calculations happens where should also be specifiable and be able to be made deterministicor should not have an effect on the result. A map reduce process' reduce step, merging results from various places also should be able to be made to give reproducible results, regardless of which results arrive first or from where.

Is our tooling too bad for this?

[TeMPOraL](https://news.ycombinator.com/user?id=TeMPOraL) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961075)

\> *Is our tooling too bad for this?*

Floating points are fundamentally too bad for this. We use them because *they're fast*, which usually more than compensates for inaccuracies FP math introduces.

(One, dealing with FP errors is mostly a fixed cost - there's a branch of CS/mathematics specializing in it, producing formally proven recipes for computing specific things in way that minimize or at least give specific bounds on errors. That's work that can be done once, and reused forever. Two, most programmers are oblivious to those issues anyway, and we've learned to live with the bugs :).)

When your parallel map-reduce is just doing matrix additions and multiplications, guaranteeing order of execution comes with serious overhead. For one, you need to have all partial results available together before reducing, so either the reduction step needs to have enough memory to store a copy of all the inputs, or it needs to block the units computing those inputs until all of them finish. Meanwhile, if you drop the order guarantee, then the reduction step just needs one fixed-size accumulator, and every parallel unit computing the inputs is free to go and do something else as soon as it's done.

So the price you pay for deterministic order is either a reduction of throughput or increase in on-chip memory, both of which end up translating to slower and more expensive hardware. The incentives strongly point towards not giving such guarantees if it can be avoided - keep in mind that GPUs have been designed for videogames (and graphics in general), and for this, floating point inaccuracies only matter when they become noticeable to the user.

[Dylan16807](https://news.ycombinator.com/user?id=Dylan16807) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960601)

Why would the same software on the same GPU architecture use different commutations from run to run?

Also if you're even considering fixed point math, you can use integer accumulators to add up your parallel chunks.

[TeMPOraL](https://news.ycombinator.com/user?id=TeMPOraL) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960879)

Why would the same multithreaded software run on the same CPU (not just architecture - the same physical chip) have its instructions execute in different order from run to run? *Performance*. Want things deterministic? You have to explicitly keep them in sync yourself. GPUs sport tens of thousands of parallel processors these days, which are themselves complex, and are linked together with more complexity, both hardware and software. They're designed to calculate *fast*, not to ensure every subprocessor is always in lock step with every other one.

Model inference on GPU is mostly doing a lot of GPU equivalent of parallelized product on (X1, X2, X3, ... Xn), where each X is itself some matrix computed by a parallelized product of other matrices. Unless there's some explicit guarantee somewhere that the reduction step will *pause until it gets all results* so it can guarantee order, instead of reducing eagerly, each such step is a non-determinism transducer, turning undetermined execution order into floating point errors via commutation.

I'm not a GPU engineer so I don't know for sure, especially about the new cards designed for AI, but since reducing eagerly allows more memory-efficient design and improves throughput, and GPUs until recently were optimized for *games* (where FP accuracy doesn't matter that much), and I don't recall any vendor making determinism a marketing point recently, I don't believe GPUs suddenly started to guarantee determinism at expense of performance.

[Dylan16807](https://news.ycombinator.com/user?id=Dylan16807) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42964003)

Each thread on a CPU will go in the same order.

Why would the reduction step of a single neuron be split across multiple threads? That sounds slower and more complex than the naive method. And if you *do* decide to write code doing that, then just the code that reduces across multiple blocks needs to use integers, so pretty much no extra effort is needed.

Like, is there a nondeterministic-dot-product instruction baked into the GPU at a low level?

[TeMPOraL](https://news.ycombinator.com/user?id=TeMPOraL) [on Feb 7, 2025](https://news.ycombinator.com/item?id=42971002)

\> *Each thread on a CPU will go in the same order.*

Not unless you control the underlying scheduler and force deterministic order; knowledge of all the code running isn't sufficient, as some factors affecting threading order are correlated with physical environment. For example, minute temperature gradient differences on the chip between two runs could affect how threads are allocated to CPU cores and order in which they finish.

\> *Why would the reduction step of a single neuron be split across multiple threads?*

Doesn't have to, but can, depending on how many inputs it has. Being able to assume commutativity gives you a lot of flexibility in how you parallelize it, and allows you to minimize overhead (both in throughput and memory requirements).

\> *Like, is there a nondeterministic-dot-product instruction baked into the GPU at a low level?*

No. There's just no dot-product instruction baked into GPU at low level that could handle vectors of arbitrary length. You need to write a loop, and that usually becomes some kind of parallel reduce.

[Dylan16807](https://news.ycombinator.com/user?id=Dylan16807) [on Feb 7, 2025](https://news.ycombinator.com/item?id=42971187)

\> could affect how threads are allocated to CPU cores and order in which they finish

I'm very confused by how you're interpreting the word "each" here.

\> Being able to assume commutativity gives you a lot of flexibility in how you parallelize it, and allows you to minimize overhead (both in throughput and memory requirements).

Splitting up a single neuron seems like something that would only increase overhead. Can you please explain how you get "a lot" of flexibility?

\> You need to write a loop, and that usually becomes some kind of parallel reduce.

Processing a layer is a loop within a loop.

The outer loop is across neurons and needs to be parallel.

The inner loop processes every weight for a single neuron and making it parallel sounds like extra effort just to increase instruction count and mess up memory locality and make your numbers less consistent.

[TeMPOraL](https://news.ycombinator.com/user?id=TeMPOraL) [on Feb 7, 2025](https://news.ycombinator.com/item?id=42971500)

I feel like you're imagining a toy network with couple dozen neurons in few layers, done on a CPU. But consider a more typical case of dozens of layers with hundreds (or thousands) of neurons each. That's some thousand numbers to reduce per each neuron.

Then, remember that GPUs are built around thousands of tiny parallel processors, each able to process a bunch (e.g. 16) parallel threads, but then the threads have to run in larger batches (SIMD-like), and there's a complex memory management architecture built-in, over which you only have so much control. Specific numbers of cores, threads, buffer sizes, as well as access patterns, differ between GPU models, and for optimal performance, you have to break down your computation to maximize utilization. Or rather, have the runtime do it for you.

This ain't an an FPGA, you don't get to organize hardware to match your network. If you have a 1000 neurons per hidden layer, then individual neurons likely won't fit on a single CUDA core, so you will have to split them down the middle, at least if you're using full-float math. Speaking of, the precision of the numbers you use is another parameter that adds to the complexity.

On the one hand, you have a bunch of mostly-linear matrix algebra, where you can tune precision. On the other hand, you have a GPU-model-specific number of parallel processors (\~thousands), that can fit only so much memory, can run some specific number of SIMD-like threads in parallel, and most of those numbers are powers of two (or a multiple of), so you have also alignment to take into account, on top of memory access patterns.

By default, your network will in no way align to any of that.

It shouldn't be hard to see that assuming commutativity gives you (or rather the CUDA compiler) much more flexibility to parallelize your calculations by splitting it whichever way it likes to maximize utilization.

[Dylan16807](https://news.ycombinator.com/user?id=Dylan16807) [on Feb 7, 2025](https://news.ycombinator.com/item?id=42976132)

I'm not imagining toy sizes. Quite the opposite. I'm saying that layers are so big that splitting per neuron already gives you a ton of individual calculations to schedule and that's plenty to get full usage out of the hardware.

You can do very wide calculations on a single neuron if you want; throwing an entire SM (64 or 128 CUDA cores) at a single neuron is trivial to do in a deterministic way. And if you have a calculation so big you benefit from splitting it across SMs, doing a deterministic sum at the end will use an unmeasurably small fraction of your runtime.

And I'll remind you that I wasn't even talking about determinism across architectures, just within an architecture, so go ahead and optimize your memory layouts and block sizes to your exact card.

[michalsustr](https://news.ycombinator.com/user?id=michalsustr) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960090)

I recently attended a STAC conference where they claimed the GPUs themselves are not deterministic. The hand-wavy speculation is they need to temperature control the cores and the flop ops may be reordered during that process. (By temperature I mean physical temperature, not some nn sampling parameter). On such large scale of computation these small differences can show up in the actually different tokens.

[fancyfredbot](https://news.ycombinator.com/user?id=fancyfredbot) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960198)

I can assure you this isn't true. Having worked with GPUs for many years in an application where consist results are important it's not only possible but actually quite easy to ensure consistent inputs produce consistent results. The temperature and clock speed do not affect the order of operations, only the speed, and this doesn't affect the results. This is the same as with any modern CPU which will also adjust clock for temperature.

[petesergeant](https://news.ycombinator.com/user?id=petesergeant) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959232)

The parent is suggesting that temperature only applies at the generation step, but the choice of backend “expert model” that a request is given to (and then performs the generation) is non-deterministic. Rather than being a single set of weights, there are a few different sets of weights that constitute the “expert” in MoE. I have no idea if that’s true, but that’s the assertion

[brookst](https://news.ycombinator.com/user?id=brookst) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959382)

I don't think it makes sense? Somewhere there has to be a RNG for that to be true. MOE itself doesn't introduce randomness, and the routing to experts is part of the model weights, not (I think) a separate model.

[pigscantfly](https://news.ycombinator.com/user?id=pigscantfly) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959524)

The samples your input is batched with on the provider's backend vary between calls and sparse mixture of experts routing when implemented for efficient utilization induces competition among tokens with either encouraged or enforced balance of expert usage among tokens in the same fixed-size group. I think it's unknown or at least undisclosed *exactly* why sequence non-determinism at zero temperature occurs in these proprietary implementations, but I think this is a good theory.

\[1\] [https://arxiv.org/abs/2308.00951](https://arxiv.org/abs/2308.00951) pg. 4 \[2\] [https://152334h.github.io/blog/non-determinism-in-gpt-4/](https://152334h.github.io/blog/non-determinism-in-gpt-4/)

[kettleballroll](https://news.ycombinator.com/user?id=kettleballroll) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959629)

I thought the temperature only affects randomness at the end of the network (when turning embeddings back I to words using the softmax). It cannot influence routing, which is inherently influenced by which examples get batched together (ie, it might depend on other users of the system)

[menaerus](https://news.ycombinator.com/user?id=menaerus) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960047)

You don't need RNG since the whole transformer is an extremely large floating-point arithmetic unit. A wild guess - how about the source of non-determinism is coming from the fact that, on the HW level, tensor execution order is not guaranteed and therefore (T0 \* T1) \* T2 can produce slightly different results than T0 \* (T1 \* T2) due to rounding errors?

[daralthus](https://news.ycombinator.com/user?id=daralthus) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961722)

I have seen numbers come differently in JAX just depending on the batch size, simply because the compiler optimizes to a different sequence of operations on the hardware.

[kiratp](https://news.ycombinator.com/user?id=kiratp) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960778)

Quantized floating point math can, under certain scenarios, be non-associative.

When you combine that fact with being part of a diverse batch of requests over an MoE model, outputs are non-deterministic.

[bushbaba](https://news.ycombinator.com/user?id=bushbaba) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42962678)

That’s why you have azure openAI APIs which give a lot more consistency

[itissid](https://news.ycombinator.com/user?id=itissid) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955824)

Wait isn't there atleast a two step process here one is semantic segmentation followed by a method like texttract for text - to avoid hallucinations?

One cannot possibly say that "Text extracted by a multimodal model cannot hallucinate"?

\> accuracy was like 96% of that of the vendor and price was significantly cheaper.

I would like to know how this 96% was tested. If you use a human to do random sample based testing, well how do you adjust the random sample for variations in distribution of errors that vary like a small set of documents could have 90% of the errors and yet they are only 1% of the docs?

[themanmaran](https://news.ycombinator.com/user?id=themanmaran) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955928)

One thing people always forget about traditional OCR providers (azure, tesseract, aws textract, etc.) is that they're \~85% accurate.

They are all probabilistic. You literally get back characters \+ confidence intervals. So when textract gives you back incorrect characters, is that a hallucination?

[kapitalx](https://news.ycombinator.com/user?id=kapitalx) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956144)

I'm the founder of [https://doctly.ai](https://doctly.ai), also pdf extraction.

The hallucination in LLM extraction is much more subtle as it will rewrite full sentences sometimes. It is much harder to spot when reading the document and sounds very plausible.

We're currently working on a version where we send the document to two different LLMs, and use a 3rd if they don't match to increase confidence. That way you have the option of trading compute and cost for accuracy.

[LeafItAlone](https://news.ycombinator.com/user?id=LeafItAlone) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959138)

\>We're currently working on a version where we send the document to two different LLMs, and use a 3rd if they don't match to increase confidence.

I’m interested to hear more about the validation process here. In my limited experience, I’ve sent the same “document” to multiple LLMs and gotten differing results. But sometimes the “right” answer was in the minority of responses. But over a large sample (same general intent of document, but very different possible formats of the information within), there was no definitive winner. We’re still working on this.

[nnurmanov](https://news.ycombinator.com/user?id=nnurmanov) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958955)

What if you use a different prompt to check the result, did this work? I am thinking to use this approach, but now I think maybe it is better to use two different LLM like you do.

[anon373839](https://news.ycombinator.com/user?id=anon373839) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956082)

It’s a question of scale. When a traditional OCR system makes an error, it’s confined to a relatively small part of the overall text. (Think of “Plastics” becoming “PIastics”.) When a LLM hallucinates, there is no limit to how much text can be made up. Entire sentences can be rewritten because the model thinks they’re more plausible than the sentences that were actually printed. And because the bias is always toward plausibility, it’s an especially insidious problem.

[themanmaran](https://news.ycombinator.com/user?id=themanmaran) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956754)

It's a bit of a pick your poison situation. You're right that traditional OCR mistakes are usually easy to catch (except when you get $30.28 vs $80.23). Compared to LLM hallucinations that are always plausibly correct.

But on the flip side, layout is often times the biggest determinant of accuracy, and that's something LLMs do a way better job on. It doesn't matter if you have 100% accurate text from a table, but all that text is balled into one big paragraph.

Also the "pick the most plausible" approach is a blessing and a curse. A good example is the handwritten form here \[1\]. GPT 4o gets the all the email addresses correct because it can reasonably guess these people are all from the same company. Whereas AWS treats them all independently and returns three different emails.

\[1\] [https://getomni.ai/ocr-demo](https://getomni.ai/ocr-demo)

[miki123211](https://news.ycombinator.com/user?id=miki123211) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959916)

The difference is the kind of hallucinations you get.

Traditional OCR is more likely to skip characters, or replace them with similar -looking ones, so you often get AL or A1 instead of AI for example. In other words, traditional spelling mistakes. LLMs can do anything from hallucinating new paragraphs to slightly changing the meaning of a sentence. The text is still grammatically correct, it makes sense in the context, except that it's not what the document actually said.

I once gave it a hand-written list of words and their definitions and asked it to turn that into flashcards (a json array with "word" and "definition"). Traditional OCR struggled with this text, the results were extremely low-quality, badly formatted but still somewhat understandable. The few LLMs I've tried either straight up refused to do it, or gave me the correct list of words, but *entirely hallucinated the definitions*.

[Scoundreller](https://news.ycombinator.com/user?id=Scoundreller) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956229)

\> You literally get back characters \+ confidence intervals.

Oh god, I wish speech to text engines would colour code the whole thing like a heat map to focus your attention to review where it may have over-enthusiastically guessed at what was said.

You no knot.

[gioazzi](https://news.ycombinator.com/user?id=gioazzi) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956688)

We did this for a speech to text solution in healthcare. Doctors would always review everything that was transcribed manually (you don’t want hallucinations in your prescription), and using a heatmap it was trivial to identify e.g. drugs that were pretty much always misunderstood by STT

[somebehemoth](https://news.ycombinator.com/user?id=somebehemoth) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956071)

I know nothing about OCR providers. It seems like OCR failure would result in gibberish or awkward wording that might be easy to spot. Doesn't the LLM failure mode assert made up truths eloquently that are more difficult to spot?

[nyarlathotep\_](https://news.ycombinator.com/user?id=nyarlathotep_) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957965)

\> is that they're \~85% accurate.

Speaking from experience, you need to double check "I" and "l" and "1" "0" and "O" all the time, accuracy seems to depend on the font and some other factors.

have a util script I use locally to copy some token values out of screenshots from a VMWare client (long story) and I have to manually adjust 9/times.

How relevant that is or isn't depends on the use case.

[itissid](https://news.ycombinator.com/user?id=itissid) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955868)

For an OCR company I imagine it is unconscionable to do this because if you would say OCR for an Oral History project for a library and you made hallucination errors, well you've replaced facts with fiction. Rewriting history? What the actual F.

[phatfish](https://news.ycombinator.com/user?id=phatfish) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956814)

Probaly totally fine for a "fintech" (Crypto?) though. Most of them are just burning VC money anyway. Maybe a lucky customer gets a windfall because Gemini added some zeros.

[rapind](https://news.ycombinator.com/user?id=rapind) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959202)

I think you can just ask DeepSeek to create a coin for you at this point, and with the recent elimination of any oversight, you can automate your rug pulls...

[threecheese](https://news.ycombinator.com/user?id=threecheese) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42962109)

Normal OCR (like Tesseract) can be wrong as well (and IMO this happens frequently). It won’t hallucinate/straight make shit up like an LLM, but a human needs to review OCR results if the workload requires accuracy. Even across multiple runs of the same image an OCR can give different results (in some scenarios). No OCR system is perfectly accurate, they all use some kind of machine learning/floating point/potentially nondeterministic tech.

[nthingtohide](https://news.ycombinator.com/user?id=nthingtohide) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958906)

Can confirm using gemini, some figure numbers were hallucinated. I had to cross-check each row to make sure data extracted is correct.

[godapi](https://news.ycombinator.com/user?id=godapi) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959636)

use different models to extract the page and cross check against each other. generally reduces issues alot

[basch](https://news.ycombinator.com/user?id=basch) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956266)

Wouldn’t the temperature on something like OCR be very low. You want the same result every time. Isn’t some part of hallucination the randomness of temperature?

[manmal](https://news.ycombinator.com/user?id=manmal) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42957028)

I can imagine reducing temp too much will lead to garbage results in situations where glyphs are unreadable.

[2rsf](https://news.ycombinator.com/user?id=2rsf) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961487)

Isn't it a good thing in this case? this is fintec, so if in doubt get a human to look at it

[basch](https://news.ycombinator.com/user?id=basch) [on Feb 10, 2025](https://news.ycombinator.com/item?id=43003306)

so you want every time you scan something illegible, for it to return a different result.

[serjester](https://news.ycombinator.com/user?id=serjester) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956685)

The LLM's are near perfect (maybe parsing I instead of 1) - if you're using the outputs in the context of RAG, your errors are likely *much much* higher in the other parts of your system. Spending a ton of time and money chasing 9's when 99% of your system's errors have totally different root causes seems like a bad use of time (unless they're not).

[j\_timberlake](https://news.ycombinator.com/user?id=j_timberlake) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957231)

This sounds extremely like my old tax accounting job. OCR existed and "worked" but it was faster to just enter the numbers manually than fix all the errors.

Also, the real solution to the problem should have been for the IRS to just pre-fill tax returns with all the accounting data that they obviously already have. But that would require the government to care.

[eternauta3k](https://news.ycombinator.com/user?id=eternauta3k) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959469)

Germany (not exactly the cradle of digitalization) already auto-fills salary tax fields with data from the employer.

[Andrex](https://news.ycombinator.com/user?id=Andrex) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957382)

They finally made filing free.

So, maybe this century?

[kennyloginz](https://news.ycombinator.com/user?id=kennyloginz) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958476)

Check again, Elon and his Doge team killed that.

[happyopossum](https://news.ycombinator.com/user?id=happyopossum) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959610)

No they didn’t, that claim is ridiculously easy to debunk but it has been going around because it fits the narrative.

[djeastm](https://news.ycombinator.com/user?id=djeastm) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42965488)

It'd be nicer if you wouldn't presume to know the reasons people might believe erroneous information.

In this case, the reason for the misinformation is do to the lack of communication from the DOGE entity regarding their actions. Mr. Musk wrote via Tweet that he had "deleted" the digital services agency "18F" that develops the IRS Free File program and also deleted their X account.

[https://apnews.com/article/irs-direct-file-musk-18f-6a4dc35a...](https://apnews.com/article/irs-direct-file-musk-18f-6a4dc35a92f9f29c310721af53f58b16)

If indeed he did cut the agency, it remains to be see how long the application will be operational.

[panarky](https://news.ycombinator.com/user?id=panarky) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953745)

This is a big aha moment for me.

If Gemini can do semantic chunking at the same time as extraction, all for so cheap and with nearly perfect accuracy, and without brittle prompting incantation magic, this is huge.

[wussboy](https://news.ycombinator.com/user?id=wussboy) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956778)

Could it do exactly the same with a web page? Would this replace something like beautiful soup?

[eitally](https://news.ycombinator.com/user?id=eitally) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959381)

I don't know exactly how or what it's doing behind the scenes, but I've been massively impressed with the results Gemini's Deep Research mode has generated, including both traditional LLM freeform & bulleted output, but also tabular data that had to come from somewhere. I haven't tried cross-checking for accuracy but the reports do come with linked sources; my current estimation is that they're at least as good as a typical analyst at a consulting firm would create as a first draft.

[fallinditch](https://news.ycombinator.com/user?id=fallinditch) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954961)

If I used Gemini 2.0 for extraction and chunking to feed into a RAG that I maintain on my local network, then what sort of locally-hosted LLM would I need to gain meaningful insights from my knowledge base? Would a 13B parameter model be sufficient?

[jhoechtl](https://news.ycombinator.com/user?id=jhoechtl) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955807)

Ypur lovalodel has littleore to do but stitch the already meaningzl pieces together.

The pre-step, chunking and semantic understanding is all that counts.

[yeahwhatever10](https://news.ycombinator.com/user?id=yeahwhatever10) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957472)

Do you get meaningful insights with current RAG solutions?

[fallinditch](https://news.ycombinator.com/user?id=fallinditch) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958403)

Yes. For example, to create AI agent 'assistants' that can leverage a local RAG in order to assist with specialist content creation or operational activities.

[potatoman22](https://news.ycombinator.com/user?id=potatoman22) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953937)

Small point but is it doing semantic chunking, or loading the entire pdf into context? I've heard mixed results on semantic chunking.

[panarky](https://news.ycombinator.com/user?id=panarky) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954142)

It loads the entire PDF into context, but then it would be my job to chunk the output for RAG, and just doing arbitrary fixed-size blocks, or breaking on sentences or paragraphs is not ideal.

So I can ask Gemini to return chunks of variable size, where each chunk is a one complete idea or concept, without arbitrarily chopping a logical semantic segment into multiple chunks.

[thelittleone](https://news.ycombinator.com/user?id=thelittleone) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955403)

Fixed size chunks is holding back a bunch of RAG projects on my backlog. Will be extremely pleased if this semantic chunking solves the issue. Currently we're getting around an 78-82% success on fixed size chunked RAG which is far too low. Users assume zero results on a RAG search equates to zero results in the source data.

[refulgentis](https://news.ycombinator.com/user?id=refulgentis) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955675)

FWIW, you might be doing it / ruled it out already:

\- BM25 to eliminate the 0 results in source data problem

\- Longer term, a peek at Gwern's recent hierarchical embedding article. Got decent early returns even with fixed size chunks

[thelittleone](https://news.ycombinator.com/user?id=thelittleone) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955786)

Much appreciated.

For others interested in BM25 for the use case above, I found this thread informative.

[https://news.ycombinator.com/item?id\=41034297](https://news.ycombinator.com/item?id=41034297)

[mediaman](https://news.ycombinator.com/user?id=mediaman) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956001)

Agree, BM25 honestly does an amazing job on its own sometimes, especially if content is technical.

We use it in combination with semantic but sometimes turn off the semantic part to see what happens and are surprised with the robustness of the results.

This would work less well for cross-language or less technical content, however. It's great for acronyms, company or industry specific terms, project names, people, technical phrases, and so on.

[jacobr1](https://news.ycombinator.com/user?id=jacobr1) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42957023)

Also consider methods that are using reasoning to potentially dispatch additional searches based on analysis of the returned data

[nnurmanov](https://news.ycombinator.com/user?id=nnurmanov) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958974)

This is my problem as well; do you have lots of documents?

[Tostino](https://news.ycombinator.com/user?id=Tostino) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956543)

I wish we had a local model for semantic chunking. I've been wanting one for ages, but haven't had the time to make a dataset and finetune that task \=/.

[hattmall](https://news.ycombinator.com/user?id=hattmall) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958312)

It's cheap now because Google is subsidizing it, no?

[vrosas](https://news.ycombinator.com/user?id=vrosas) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958391)

Spoiler: every model is deeply, deeply subsidized. At least Google's is subsidized by a real business with revenue, not VC's staring at the clock.

[panarky](https://news.ycombinator.com/user?id=panarky) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958952)

It's cheap because it's a Flash model, far smaller and much less compute for inference, runs on TPUs instead of GPUs.

[ForHackernews](https://news.ycombinator.com/user?id=ForHackernews) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956650)

This is great, I just want to highlight out how nuts it is that we have spun up whole industries around extracting text that was typically printed from a computer, back into a computer.

There should be laws that mandates that financial information be provided in a sensible format: even Office Open XML would be better than this insanity. Then we can redirect all this wasted effort into digging ditches and filling them back in again.

[faxmeyourcode](https://news.ycombinator.com/user?id=faxmeyourcode) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953799)

I've been fighting trying to chunk SEC filings properly, specifically surrounding the strange and inconsistent tabular formats present in company filings.

This is giving me hope that it's possible.

[anirudhb99](https://news.ycombinator.com/user?id=anirudhb99) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954096)

(from the gemini team) we're working on it! semantic chunking & extraction will definitely be possible in the coming months.

[otoburb](https://news.ycombinator.com/user?id=otoburb) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953909)

\>>*I've been fighting trying to chunk SEC filings properly, specifically surrounding the strange and inconsistent tabular formats present in company filings.*

For this specific use case you can also try edgartools\[1\] which is a library that was relatively recently released that ingests SEC submissions and filings. They don't use OCR but (from what I can tell) directly parse the XBRL documents submitted by companies and stored in EDGAR, if they exist.

\[1\] [https://github.com/dgunning/edgartools](https://github.com/dgunning/edgartools)

[faxmeyourcode](https://news.ycombinator.com/user?id=faxmeyourcode) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956604)

I'll definitely be looking into this, thanks for the recommendation! Been playing around with it this afternoon and it's very promising.

[barrenko](https://news.ycombinator.com/user?id=barrenko) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953991)

If you'd kindly tl;dr the chunking strategies you have tried and what works best, I'd love to hear.

[jgalt212](https://news.ycombinator.com/user?id=jgalt212) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954491)

isn't everyone on iXBRL now? Or are you struggling with historical filings?

[faxmeyourcode](https://news.ycombinator.com/user?id=faxmeyourcode) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956278)

XBRL is what I'm using currently, but it's still kind of a mess (maybe I'm just bad at it) for some of the non-standard information that isn't properly tagged.

[yzydserd](https://news.ycombinator.com/user?id=yzydserd) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954088)

How do today’s LLM’s like Gemini compare with the Document Understanding services google/aws/azure have offered for a few years, particularly when dealing with known forms? I think Google’s is Document AI.

[zacmps](https://news.ycombinator.com/user?id=zacmps) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954334)

I've found the highest accuracy solution is to OCR with one of the dedicated models then feed that text and the original image into an LLM with a prompt like:

"Correct errors in this OCR transcription".

[therein](https://news.ycombinator.com/user?id=therein) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954985)

How does it behave if the body of text is offensive or what if it is talking about a recipe to purify UF-6 gas at home? Will it stop doing what it is doing and enter lecturing mode?

I am asking not to be cynical but because of my limited experience with using LLMs for any task that may operate on offensive or unknown input seems to get triggered by all sorts of unpredictable moral judgements and dragged into generating not the output I wanted, at all.

If I am asking this black box to give me a JSON output containing keywords for a certain text, if it happens to be offensive, it refuses to do that.

How does one tackle with that?

[devjab](https://news.ycombinator.com/user?id=devjab) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959036)

We use the Azure models and there isn't an issue with safety filters as such for enterprise customers. The one time we had an issue microsoft changed the safety measures. Of course the safety measures we might meet are the sort of engineering which could be interpreted as weapons manufacturing, and not "political" as such. Basically the safety guard rails seem to be added on top of all these models, which means they can also be removed without impacting the model. I could be wrong on that, but it seems that way.

[xnx](https://news.ycombinator.com/user?id=xnx) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955555)

There are many settings for changing the safety level in Gemini API calls: [https://ai.google.dev/gemini-api/docs/safety-settings](https://ai.google.dev/gemini-api/docs/safety-settings)

[shijithpk](https://news.ycombinator.com/user?id=shijithpk) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961586)

This is for anyone coming across this link later. In their latest SDKs, if you want to completely switch off their safety settings, the flag to use is 'OFF' and not 'BLOCK\_NONE' as mentioned in the docs in the link above.

The Gemini docs don't refect that change yet. [https://discuss.ai.google.dev/t/safety-settings-2025-update-...](https://discuss.ai.google.dev/t/safety-settings-2025-update-broken-again/59360/25)

[sumedh](https://news.ycombinator.com/user?id=sumedh) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955905)

Try setting the safety params to none and see if that makes any difference.

[zacmps](https://news.ycombinator.com/user?id=zacmps) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955472)

It's not something I've needed to deal with personally.

We have run into added content filters in Azure OpenAI on a different application, but we just put in a request to tune them down for us.

[bradfox2](https://news.ycombinator.com/user?id=bradfox2) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954503)

This is what we do today. Have you tried it against Gemini 2.0?

[anirudhb99](https://news.ycombinator.com/user?id=anirudhb99) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956923)

member of the gemini team here -- personally, i'd recommend directly using gemini vs the document understanding services for OCR & general docs understanding tasks. From our internal evals gemini is now stronger than these solutions and is only going to get much better (higher precision, lower hallucination rates) from here.

[joelhaus](https://news.ycombinator.com/user?id=joelhaus) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958486)

Could we connect offline about using Gemini instead of the doc ai custom extractor we currently use in production?

This sounds amazing & I'd love your input on our specific use case.

joelatoutboundin.com

[ajcp](https://news.ycombinator.com/user?id=ajcp) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955867)

GCP's Document AI service is now literally just a UI layer specific to document parsing use-cases back by Gemini models. When we realized that we dumped it and just use Gemini directly.

[xnx](https://news.ycombinator.com/user?id=xnx) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955520)

Your OCR vendor would be smart to replace their own system with Gemini.

[TeMPOraL](https://news.ycombinator.com/user?id=TeMPOraL) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960661)

They will, and they'll still have a solid product to sell, because their value proposition isn't accurate OCR per se, but *putting an SLA on it*.

Reaching reliability with LLM OCR might involve some combination of multiple LLMs (and keeping track of how they change), perhaps mixed with old-school algorithms, and *random sample reviews by humans*. They can tune this pipeline however they need at their leisure to eke out extra accuracy, and then put written guarantees on top, and still be cheaper for you long-term.

[\_the\_inflator](https://news.ycombinator.com/user?id=_the_inflator) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955673)

With “Next generation, extremely sophisticated AI” to be precise, I wait say. ;)

Marketing joke aside, maybe a hybrid approach could serve the vendor well. Best of both worlds if it reaps benefits or even have a look at hugging face for even more specialized aka better LLMs.

[martingoodson](https://news.ycombinator.com/user?id=martingoodson) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961030)

I work in financial data and our customers would not accept 96% accuracy in the data points we supply. Maybe 99.96%.

For most use cases in financial services, accurate data is very important.

[riku\_iki](https://news.ycombinator.com/user?id=riku_iki) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42967300)

so, what solution are you using to extract data with 99.96% accuracy?

[llm\_bot](https://news.ycombinator.com/user?id=llm_bot) [on Feb 10, 2025](https://news.ycombinator.com/item?id=43000649)

I'm curious to hear about your experience with this. Which solution were you using before (the one that took 12 minutes)? If it was a self-hosted solution, what hardware were you using? How does Gemini handle PDFs with an unknown schema, and how does it compare to other general PDF parsing tools like Amazon Textract or Azure Document Intelligence? In my initial test, tables and checkboxes weren't well recognized.

[eru](https://news.ycombinator.com/user?id=eru) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958869)

\> For the 4% inaccuracies a lot of them are things like the text "LLC" handwritten would get OCR'd as "IIC" which I would say is somewhat "fair".

I'm actually somewhat surprised Gemini didn't guess from context that LLC is much more likely?

I guess the OCR subsystem is intentionally conservative? (Though I'm sure you could do a second step on your end, take the output from the conservative OCR pass, and sent it through Gemini and ask it to flag potential OCR problems? I bet that would flag most of them with very few false positives and false negatives.)

[PaulHoule](https://news.ycombinator.com/user?id=PaulHoule) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42964161)

Where I work we've had great success at using LLMs to OCR paper documents that look like

[https://static.foxnews.com/foxnews.com/content/uploads/2023/...](https://static.foxnews.com/foxnews.com/content/uploads/2023/12/Fox_December-10-13-2023_National_Topline_December-17-Release.pdf)

but were often written with typewriters long ago to get nice structured tabular output. Deals with text being split across lines and across pages just fine.

[chewkokwah](https://news.ycombinator.com/user?id=chewkokwah) [on Feb 9, 2025](https://news.ycombinator.com/item?id=42989066)

How about the comparison with traditional proprietary on premise software like ONMIPage or ABBYY or those listed below: [https://en.wikipedia.org/wiki/Comparison\_of\_optical\_characte...](https://en.wikipedia.org/wiki/Comparison_of_optical_character_recognition_software)

[chickenWing](https://news.ycombinator.com/user?id=chickenWing) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42966080)

It is cheaper now, but I wonder if it will continue to be cheaper when companies like Google and OpenAI decide they want to make a profit off of AI, instead of pouring billions of dollars of investment funds into it. By the time that happens, many of the specialized service providers will be out of business and Google will be free to jack up the price.

[kbaker](https://news.ycombinator.com/user?id=kbaker) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42966138)

I use Claude through OpenRouter (with Aider), and was pretty amazed to see that it routes the requests during the same session almost round-robin through Amazon Bedrock, sometimes through Google Vertex, sometimes through Anthropic themselves, all of course using the same underlying model.

Literally whoever has the cheapest compute.

With the speed that AI models are improving these days, it seems like the 'moat' of a better model is only a few months before it is commoditized and goes to the cheapest provider.

[mnky9800n](https://news.ycombinator.com/user?id=mnky9800n) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958152)

What are the pdfs containing?

I’ve been wanting to build a system that ingests pdf reports that reference other types of data like images, csv, etc. that can also be ingested to ultimately build an analytics database from the stack of unsorted data AB’s meta data but I have not found any time to do anything like that yet. What kind of tooling do you use to build your data pipelines?

[eitally](https://news.ycombinator.com/user?id=eitally) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959364)

It's great to hear it's this good, and it makes sense since Google has had several years of experience creating document-type-specific OCR extractors as components of their Document AI product in Cloud. What most heartening is to hear that the legwork they did for that set of solutions has made it into Gemini for consumers (and businesses).

[RamblingCTO](https://news.ycombinator.com/user?id=RamblingCTO) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959695)

Successful document processing vendors to use LLMs already. I know this at least of klippa. They have (apparently) fine-tuned models, prompts etc. The biggest issue with using LLMs directly is error handling, validation and "parameter drift"/randomness. This is the typical "I'll build it myself but worse" thing

[octonaut](https://news.ycombinator.com/user?id=octonaut) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958534)

I'm interested to hear what your experience has been dealing with optional data. For example if the input pdf has fields which are sometimes not populated or nonexistent, is Gemini smart enough to leave those fields blank in the output schema? Usually the LLM tries to please you and makes up values here.

[surfingdino](https://news.ycombinator.com/user?id=surfingdino) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961554)

You could ingest them with AWS Textract and have predictability and formatting in the format of your choice. Using LLMs for this is lazy and generates unpredictable and non-deterministic results.

[faebi](https://news.ycombinator.com/user?id=faebi) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959887)

Did you try other vision models such as ChatGPT and Grok? I'm doing something similar but struggled to find good comparisons in between the vision models in terms OCR and document understanding.

[andai](https://news.ycombinator.com/user?id=andai) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960954)

If the documents have the same format, maybe you could include an example document in the prompt, so the boilerplate stuff (like LLC) gets handled properly.

[lebimas](https://news.ycombinator.com/user?id=lebimas) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42965420)

You could probably take this a step further and pipe the OCR'ed text into Claude 3.5 Sonnet and get it to fix any OCR errors

[bdd8f1df777b](https://news.ycombinator.com/user?id=bdd8f1df777b) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957905)

What if you prompt Gemini that mistaking LLC for IIC is a common mistake? Will Gemini auto correct it?

[tomrod](https://news.ycombinator.com/user?id=tomrod) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957916)

With lower temperature, it seems to work okay for me.

A \_killer\_ awesome thing it does too is allow code specification in the config instead of through repeated attempts at prompts.

[ensocode](https://news.ycombinator.com/user?id=ensocode) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960847)

Just to make sure: you are talking about your experiences with Gemini 1.5 Flash here, right?

[mring33621](https://news.ycombinator.com/user?id=mring33621) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42963981)

Hi! Any guesstimate for pages/minute from your Gemini OCR experience? Thanks!

[depr](https://news.ycombinator.com/user?id=depr) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954472)

So are you mostly processing PDFs with data? Or PDFs with just text, or images, graphs?

[thelittleone](https://news.ycombinator.com/user?id=thelittleone) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955428)

Not the parent, but we process PDFs with text, tables, diagrams. Works well if the schema is properly defined.

[\_blk](https://news.ycombinator.com/user?id=_blk) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957624)

Is privacy a concern?

[surfingdino](https://news.ycombinator.com/user?id=surfingdino) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961563)

Why would it be? Their only concern is IPO.

[jack\_pp](https://news.ycombinator.com/user?id=jack_pp) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959026)

In fintech I'd suspect the PDFs are public knowledge

[cess11](https://news.ycombinator.com/user?id=cess11) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953680)

What hardware are you using to run it?

[kccqzy](https://news.ycombinator.com/user?id=kccqzy) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954078)

The Gemini model isn't open so it does not matter what hardware you have. You might have confused Gemini with Gemma.

[cess11](https://news.ycombinator.com/user?id=cess11) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955987)

OK, I see, pity. I'm interested in similar applications but in contexts where the material is proprietary and might contain PII.

[rco8786](https://news.ycombinator.com/user?id=rco8786) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42962009)

“LLC” to “IIC” is one thing. But wouldn’t that also make it just as easy to to mistake something like “$100” for “$700”?

[sensecall](https://news.ycombinator.com/user?id=sensecall) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955083)

Out of interest, did you parse into any sort of defined schema/structure?

[gnat](https://news.ycombinator.com/user?id=gnat) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955159)

Parent literally said so …

\> Our prompt is currently very simple: "OCR this PDF into this format as specified by this json schema" and didn't require some fancy "prompt engineering" to contort out a result.

[bionhoward](https://news.ycombinator.com/user?id=bionhoward) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955470)

The Gemini api has a customer noncompete, so it’s not an option for AI, what are you working on that doesn’t compete with AI?

[B-Con](https://news.ycombinator.com/user?id=B-Con) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955529)

You do realize most people aren't working on AI, right?

Also, OP mentioned fintech at the outset.

[novaleaf](https://news.ycombinator.com/user?id=novaleaf) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955574)

what doesn't compete with ai?

[llm\_trw](https://news.ycombinator.com/user?id=llm_trw) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955414)

This is using exactly the wrong tools at every stage of the OCR pipeline, and the cost is astronomical as a result.

You don't use multimodal models to extract a wall of text from an image. They hallucinate constantly the second you get past perfect 100% high-fidelity images.

You use an object detection model trained on documents to find the bounding boxes of each document section as \_images\_; each bounding box comes with a confidence score for free.

You then feed each box of text to a regular OCR model, also gives you a confidence score along with each prediction it makes.

You feed each image box into a multimodal model to describe what the image is about.

For tables, use a specialist model that does nothing but extract tables—models like GridFormer that aren't hyped to hell and back.

You then stitch everything together in an XML file because Markdown is for human consumption.

You now have everything extracted with flat XML markup for each category the object detection model knows about, along with multiple types of probability metadata for each bounding box, each letter, and each table cell.

You can now start feeding this data programmatically into an LLM to do \_text\_ processing, where you use the XML to control what parts of the document you send to the LLM.

You then get chunking with location data and confidence scores of every part of the document to put as meta data into the RAG store.

I've build a system that read 500k pages \_per day\_ using the above completely locally on a machine that cost $20k.

[ajcp](https://news.ycombinator.com/user?id=ajcp) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956087)

Not sure what service you're basing your calculation on but with Gemmini I've processed 10,000,000\+ shipping documents (PDF and PNGs) of every concievable layout in one month at under $1000 and an accuracy rate of between 80-82% (humans were at 66%).

The longest part of the development timeline was establishing the accuracy rate and the ingestion pipeline, which itself is massively less complex than what your workflow sounds like: PDF -> Storage Bucket -> Gemini -> JSON response -> Database

Just to get sick with it we actually added some recusion to the Gemini step to have it rate how well it extracted, and if it was below a certain rating to rewrite its own instructions on how to extract the information and then feed it back into itself. We didn't see any improvement in accuracy, but it was still fun to do.

[llm\_trw](https://news.ycombinator.com/user?id=llm_trw) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956929)

\>Not sure what service you're basing your calculation on but with Gemmini

The table of costs in the blog post. At 500,000 pages per day the hardware fixed cost overcomes the software variable cost at day 240 and from then on you're paying an extra \~$100 per day to keep it running in the cloud. The machine also had to use extremely beefy GPUs to fit all the models it needed to. Compute utilization was between 5 to 10% which means that it's future proof for the next 5 years at the rate at which the data source was growing.

```
| Model                       | Pages per Dollar |
    |-----------------------------+------------------|
    | Gemini 2.0 Flash            | ≈ 6,000          |
    | Gemini 2.0 Flash Lite       | ≈ 12,000*        |
    | Gemini 1.5 Flash            | ≈ 10,000         |
    | AWS Textract                | ≈ 1,000          |
    | Gemini 1.5 Pro              | ≈ 700            |
    | OpenAI 4o-mini              | ≈ 450            |
    | LlamaParse                  | ≈ 300            |
    | OpenAI 4o                   | ≈ 200            |
    | Anthropic claude-3-5-sonnet | ≈ 100            |
    | Reducto                     | ≈ 100            |
    | Chunkr                      | ≈ 100            |
```

 There is also the fact that it's \_completely\_ local. Which meant we could throw in every proprietary data source that couldn't leave the company at it.

\>The longest part of the development timeline was establishing the accuracy rate and the ingestion pipeline, which itself is massively less complex than what your workflow sounds like: PDF -> Storage Bucket -> Gemini -> JSON response -> Database

Each company should build tools which match the skill level of their developers. If you're not comfortable training models locally with all that entails off the shelf solutions allow companies to punch way above their weight class in their industry.

[serjester](https://news.ycombinator.com/user?id=serjester) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42957126)

That assumes that you're able to find a model that can match Gemini's performance - I haven't come across anything that comes close (although hopefully that changes).

[jeswin](https://news.ycombinator.com/user?id=jeswin) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42962443)

Nice article, mirrors my experience. Last year (around when multimodal 3.5 Sonnet launched), I had run a sizeable number of PDFs through it. Accuracy was remarkably high (99%-ish), whereas GPT was just unusable for this purpose.

[cpursley](https://news.ycombinator.com/user?id=cpursley) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956215)

Very cool! How are you storing it to a database - vectors? What do you do with the extracted data (in terms of being able to pull it up via some query system)?

[ajcp](https://news.ycombinator.com/user?id=ajcp) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956506)

In this use-case the customer just wanted data not currently in the warehouse inventory management system capatured, so here we converted a JSON response to a classic table row schema (where 1 row \= 1 document) and now boom, shipping data!

However we do very much recommend storing the raw model responses for audit and then at least as vector embeddings to orient the expectation that the data will need to be utilized for vector search and RAG. Kind of like "while we're here why don't we do what you're going to want to do at some point, even if it's not your use-case now..."

[rofl123](https://news.ycombinator.com/user?id=rofl123) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42962983)

\> Kind of like "while we're here why don't we do what you're going to want to do at some point, even if it's not your use-case now..."

wow, this is so bad. why do it now and introduce complexity and debt if you can do it later when you actually need it? you are just riding the hype wave and trying to get most out of it but that's fine.

[ajcp](https://news.ycombinator.com/user?id=ajcp) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42964940)

\> why do it now and introduce complexity and debt if you can do it later when you actually need it?

The same reason I don't wait until it snows to buy snowboots. I know my environment, topography, scale, risk-profile, and costs, and can concieve of innumerable use-cases for when they will be necessary, even if it's only May, when snowboots happen to be on sale ;) What's a little closet space and the burden of locking my door when I leave the house in the interim?

[svieira](https://news.ycombinator.com/user?id=svieira) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42965774)

\> \[with\] an accuracy rate of between 80-82% (humans were at 66%)

Was this human-verified in some way? If not, how did you establish the facts-on-the-ground about accuracy?

[ajcp](https://news.ycombinator.com/user?id=ajcp) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42967046)

Yup, unfortunately the only way to know how good an AI is at anything is to do the same way you'd do with a human: build a test that you know the answers to already. That's also why the accuracy evaluation was by far the most time intensive part of the development pipeline as we had to manually build a "ground-truth" dataset that we could evaluate the AI again.

[jeswin](https://news.ycombinator.com/user?id=jeswin) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42962387)

I feel compelled to reply. You've made a bunch of assumptions, and presented your success (likely with a limited set of table formats) as the one true way to parse PDFs. There's no such thing.

In real world usage, many tables are badly misaligned. Headers are off. Lines are missing between rows. Some columns and rows are separated by colors. Cells are merged. Some are imported from Excel. There are dotted sub sections, tables inside cells etc. Claude (and now Gemini) can parse complex tables and convert that to meaningful data. Your solution will likely fail, because rules are fuzzy in the same way written language is fuzzy.

Recently someone posted this on HN, it's a good read: [https://lukaspetersson.com/blog/2025/bitter-vertical/](https://lukaspetersson.com/blog/2025/bitter-vertical/)

\> You don't use multimodal models to extract a wall of text from an image. They hallucinate constantly the second you get past perfect 100% high-fidelity images.

No, not like that, but often as nested Json or Xml. For financial documents, our accuracy was above 99%. There are many ways to do error checking to figure out which ones are likely to have errors.

\> This is using exactly the wrong tools at every stage of the OCR pipeline, and the cost is astronomical as a result.

One should refrain making statements about cost without knowing how and where it'll be used. When processing millions of PDFs, it could be a problem. When processing 1000, one might prefer Gemini/other over spending engineering time. There are many apps where processing a single doc is say $10 in revenue. You don't care about OCR costs.

\> I've build a system that read 500k pages \_per day\_ using the above completely locally on a machine that cost $20k.

The author presented techniques which worked for them. It may not work for you, because there's no one-size-fits-all for these kinds of problems.

[metadat](https://news.ycombinator.com/user?id=metadat) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42964167)

Related discussion:

*AI founders will learn the bitter lesson*

[https://news.ycombinator.com/item?id\=42672790](https://news.ycombinator.com/item?id=42672790) - 25 days ago, 263 comments

The HN discussion contains a lot of interesting ideas, thanks for the pointer!

[llm\_trw](https://news.ycombinator.com/user?id=llm_trw) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42967331)

You're making an even less charitable set of assumptions:

1). I'm incompetent enough to ignore publicly available table benchmarks.

2). I'm incompetent enough to never look at poor quality data.

3). I'm incompetent enough to not create a validation dataset for all models that were available.

Needless to say you're wrong on all three.

My day rate is $400 \+ taxes per hour if you want to be run through each point and why VLMs like Gemini fail spectacularly and unpredictably when left to their own devices.

[pkkkzip](https://news.ycombinator.com/user?id=pkkkzip) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42967379)

whoa, this is a really aggressive response. No one is calling you incompetent rather criticizing your assumptions.

\> My day rate is $400 \+ taxes per hour if you want to be run through each point

Great, thanks for sharing.

[danielparsons](https://news.ycombinator.com/user?id=danielparsons) [on Feb 13, 2025](https://news.ycombinator.com/item?id=43031954)

bragging about billing $400 an hour LOL

[vikp](https://news.ycombinator.com/user?id=vikp) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956619)

Marker ([https://www.github.com/VikParuchuri/marker](https://www.github.com/VikParuchuri/marker)) works kind of like this. It uses a layout model to identify blocks and processes each one separately. The internal format is a tree of blocks, which have arbitrary fields, but can all render to html. It can write out to json, html, or markdown.

I integrated gemini recently to improve accuracy in certain blocks like tables. (get initial text, then pass to gemini to refine) Marker alone works about as well as gemini alone, but together they benchmark much better.

[llm\_trw](https://news.ycombinator.com/user?id=llm_trw) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42957047)

I used sxml \[0\] unironically in this project extensively.

The rendering step for reports that humans got to see was a call to pandoc after the sxml was rendered to markdown - look ma we support powerpoint! - but it also allowed us to easily convert to whatever insane markup a given large (or small) language model worked best with on the fly.

\[0\] [https://en.wikipedia.org/wiki/SXML](https://en.wikipedia.org/wiki/SXML)

[cma](https://news.ycombinator.com/user?id=cma) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960463)

Why process separately, if there are ink smudges, photocopier glitches, etc. wouldn't it guess some stuff better from richer context, like acronyms in rows used across the other tables?

[hackernewds](https://news.ycombinator.com/user?id=hackernewds) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960378)

It's funny you astroturf your own project in a thread where another is presenting tangential info about their own

[alemos](https://news.ycombinator.com/user?id=alemos) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960648)

what does marker add on top of docling?

[vikp](https://news.ycombinator.com/user?id=vikp) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42964025)

Docling is a great project, happy to see more people building in the space.

Marker output will be higher quality than docling output across most doc types, especially with the --use\_llm flag. A few specific things we do differently:

```
- We have hybrid mode with gemini that merges tables across pages, improves quality on forms, etc.
  - we run an ordering model, so ordering is better for docs where the PDF orde ris bad
  - OCR is a lot better, we train our own model, surya - https://github.com/VikParuchuri/surya
  - References and links
  - Better equation conversion (soon including inline)
```

[anon373839](https://news.ycombinator.com/user?id=anon373839) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958962)

This is a great comment. I will mention another benefit to this approach: the same pipeline works for PDFs that are digital-native and don't require OCR. After the object detection step, you collect the text directly from within the bounding boxes, and the text is error-free. Using Gemini means that you give this up.

[siva7](https://news.ycombinator.com/user?id=siva7) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958781)

You‘re describing yesterdays world. With the advancement of AI, there is no need for any of these many steps and stages of OCR anymore. There is no need for XML in your pipeline because Markdown is now equally suited for machine consumption by AI models.

[llm\_trw](https://news.ycombinator.com/user?id=llm_trw) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959526)

The results we got 18 months ago are still better than the current gemini benchmarks at a fraction the cost.

As for markdown, great. Now how do you encode the meta data about the confidence of the model that the text says what it thinks it says? Becuase xml has this lovely thing called attributes that let's you keep a provenance record without a second database that's also readable by the llm.

[JohnKemeny](https://news.ycombinator.com/user?id=JohnKemeny) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960042)

Just commenting here so that I can find back to this comment later. You perfectly captured the AI hype in one small paragraph.

[fransje26](https://news.ycombinator.com/user?id=fransje26) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960857)

Hey, why settle for yesteryear's world, with better accuracy, lower costs and local deployment, if you can use today's new shinny tool, reinvent the wheel, put everything in the cloud, and get hallucination for free..

[BenGosub](https://news.ycombinator.com/user?id=BenGosub) [on Feb 7, 2025](https://news.ycombinator.com/item?id=42970939)

What are the tools from the yesterday's world you are referring to? I've had issues with the base Python library in PDF parsing, only some state of the art tools were able to parse the information correctly.

[raincole](https://news.ycombinator.com/user?id=raincole) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961253)

Just commenting here to say the GP is spot on.

If you already have a high optimized pipeline built *yesterday*, then sure keep using it.

But if you start dealing with PDF today, just use Gemini. Use the most human readable formats you can find because we know AI will be optimized on understanding that. Don't even think about "stitching XML files" blahblah.

[aiono](https://news.ycombinator.com/user?id=aiono) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42962037)

Except it's more expensive, hallucinates and you are vendor locked.

[bitdribble](https://news.ycombinator.com/user?id=bitdribble) [on Feb 16, 2025](https://news.ycombinator.com/item?id=43069374)

Why do you say you are vendor locked? There are 4-5 top of the line LLMs that support structured output and compete with Gemini. Once an LLM vendor has the pipeline built for structured output, they'll pass each new model through the pipeline.

[tzs](https://news.ycombinator.com/user?id=tzs) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42962399)

For future reference if you click on the timestamp of a comment that will bring you to a screen that has a “favorite” link. Click that to add the comment to your favorite comments list, which you can find on your profile page.

[senko](https://news.ycombinator.com/user?id=senko) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960927)

\> I've build a system that read 500k pages \_per day\_ using the above completely locally on a machine that cost $20k.

That is impressive. However, if someone needs to read a couple of hundred pages per day, there's no point in setting all that up.

Also, you neglected to mention the cost of setting everything up. The machine cost $20k; but your time, and cost to train yolo8, probably cost more than that. If you want to compare costs (find a point where local implementation such as this is better ROI), you should compare fully loaded costs.

[thiht](https://news.ycombinator.com/user?id=thiht) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961613)

Or, depending on your use case, you do it in one step and ask an LLM to extract data from a PDF.

What you describe is obviously better and more robust by a lot, but the LLM only approach is not "wrong". It’s simple, fast, easy to setup and understand, and it *works*. With less accuracy but it does work. Depending on the constraints, development budget and load it’s a perfectly acceptable solution.

We did this to handle 2000 documents per month and are satisfied with the results. If we need to upgrade to something better in the future we will, but in the mean time, it’s done.

[eitally](https://news.ycombinator.com/user?id=eitally) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959394)

Fwiw, I'm not convinced Gemini isn't using an document-based objection detection model for this, at least some parts of this or for some doc categories (especially common things like IDs, bills, tax forms, invoices & POs, shipping documents, etc that they've previously created document extractors for (as part of their DocAI cloud service).

[simonw](https://news.ycombinator.com/user?id=simonw) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959434)

I don't see why they would do that. The whole point of training a model like Gemini is that you *train the model* - if they want it to work great against those different categories of document the likely way to do it is to add a whole bunch of those documents to Gemini's regular training set.

[woah](https://news.ycombinator.com/user?id=woah) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957414)

Getting "bitter lesson" vibes from this post

[llm\_trw](https://news.ycombinator.com/user?id=llm_trw) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958009)

The bitter lesson is very little of the sort.

If we had unlimited memory, compute and data we'd use a rank N tensor for an input of length N and call it a day.

Unfortunately N\^N grows rather fast and we have to do all sorts of interesting engineering to make ML calculations complete before the heat death of the universe.

[woah](https://news.ycombinator.com/user?id=woah) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958874)

\> Most AI research has been conducted as if the computation available to the agent were constant (in which case leveraging human knowledge would be one of the only ways to improve performance) but, over a slightly longer time than a typical research project, massively more computation inevitably becomes available. Seeking an improvement that makes a difference in the shorter term, researchers seek to leverage their human knowledge of the domain, but the only thing that matters in the long run is the leveraging of computation. These two need not run counter to each other, but in practice they tend to. Time spent on one is time not spent on the other. There are psychological commitments to investment in one approach or the other. And the human-knowledge approach tends to complicate methods in ways that make them less suited to taking advantage of general methods leveraging computation.

[llm\_trw](https://news.ycombinator.com/user?id=llm_trw) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959570)

To solve mnist without mathematical tricks like convolutions or attention heads you would nees 2.5e42 weights. Assuming that you're using 16 bit weights that 5e42 bytes. A yotta byte is 10e24.

That is you'd need 5 exa yotta bytes to solve it.

Currently the whole world has around 200 zetabytes of storage.

I short for the next 120 years mnist will need mathematical tricks to be solved.

[flask\_manager](https://news.ycombinator.com/user?id=flask_manager) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960533)

The distinction that i think is important to make when talking about "the bitter lesson" is that improving the compute and training infrastructure and tricks in the abstract wins over intelligent model and system design.

Its more about the information about the specific problem you are solving having less impact than techniques that target the compute. So in this case, breaking down how to parse a PDF in stages for your domain is involving specific expert knowledge of the domain, but training with attention is about efficient use of compute in general; with no domain expertise.

[pkkkzip](https://news.ycombinator.com/user?id=pkkkzip) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958128)

I think you are being pedantic here and business decisions aren't made based on purely cost but brittleness, maintenance, time to market.

You are assuming you can match Gemini's performance, Google's engineering resources and costs being constant in to the future.

[llm\_trw](https://news.ycombinator.com/user?id=llm_trw) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958255)

\>You are assuming you can match Gemini's performance

I'm not assuming. We already did, 18 months ago with better performance than the current generation of Gemini for our use case.

You're falling into the usual trap of thinking that because big tech spends big money it gets big results. It doesn't. To quote a friend who was a manager at google "If only I could get my team of 100 to be as productive as my first team of three.".

[klaussilveira](https://news.ycombinator.com/user?id=klaussilveira) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42962243)

Only thing I could find about GridFormer and tables was this: [https://arxiv.org/pdf/2309.14962v1](https://arxiv.org/pdf/2309.14962v1)

But there is no GitHub link or details on the implementation. Only model available seems to be one for removing weather effects from images: [https://github.com/TaoWangzj/GridFormer](https://github.com/TaoWangzj/GridFormer)

Could you care to expand on how you would use GridFormer for extracting tables from images? Seems like it's not as trivial as using something like Excalibur or Tabula, both which seem more battle-tested.

[looofooo0](https://news.ycombinator.com/user?id=looofooo0) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961296)

That sounds like a sound approach. Are the steps easliy upgradable with better models? Also it sounds like you can use an character recognition model on single characters? Do you do extra checks for numerical characters?

[Ostatnigrosh](https://news.ycombinator.com/user?id=Ostatnigrosh) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42965540)

This is exactly the wrong mentality to have about new technology.

[dr\_kiszonka](https://news.ycombinator.com/user?id=dr_kiszonka) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956265)

Impressive. Can you share anything more about this project? 500k pages a day is massive and I can imagine why one would require that much throughput.

[llm\_trw](https://news.ycombinator.com/user?id=llm_trw) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956947)

It was a financial company that needed a tool that would out perform Bloomberg terminal for traders and quants in markets where their coverage is spotty.

[thegabriele](https://news.ycombinator.com/user?id=thegabriele) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960744)

You mentioned Grid Former, i found a paper describing it (Grid Former: Towards Accurate Table Structure Recognition via Grid Prediction). How did you implemented it?

[polote](https://news.ycombinator.com/user?id=polote) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956247)

Do you know another model than gridformer to detect table that has an available implementation somewhere ?

[llm\_trw](https://news.ycombinator.com/user?id=llm_trw) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956980)

We had to roll our own from research papers unfortunately.

The number one take away we got was to use much larger images than anything that anyone else ever mentioned to get good results. A rule of thumb was that if you print the png of the image it should be easily readable from 2m away.

The actual model is proprietary and stuck in corporate land forever.

[vagab0nd](https://news.ycombinator.com/user?id=vagab0nd) [on Feb 8, 2025](https://news.ycombinator.com/item?id=42983927)

I honestly can't tell if you are being serious. Is there any doubt that the "OCR pipeline" will just be an LLM and it's just a matter of time?

What you are describing is similar to how computer used to detect cats. You first extract edges, texture and gradient. Then use a sliding window and run a classifier. Then you use NMS to merge the bounding boxes.

[ck\_one](https://news.ycombinator.com/user?id=ck_one) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955515)

What object detection model do you use?

[a3w](https://news.ycombinator.com/user?id=a3w) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955601)

Is tesseract even ML based? Oh, this piece of software is more than 19 years old, perhaps there are other ways to do good, cheap OCR now. Does Gemini have an OCR library, internally? For other LLMs, I had the feeling that the LLM scripts a few lines of python to do the actual heavy lifting with a common OCR framework.

[llm\_trw](https://news.ycombinator.com/user?id=llm_trw) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955628)

Custom trained yolo v8. I've moved on since then and the work was done in 2023. You'd get better results for much less today.

[twelve40](https://news.ycombinator.com/user?id=twelve40) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956704)

Isn't it amazing how one company invented a universally spread format that takes structured data from an editor (except images obviously) and converts it into a completely fucked-up unstructured form that then requires expensive voodoo magic to convert back into structured data.

[sconeguy](https://news.ycombinator.com/user?id=sconeguy) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956861)

What would it have taken to store the plain text in some meta field in the document. Argh, so annoying.

[dkjaudyeqooe](https://news.ycombinator.com/user?id=dkjaudyeqooe) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957625)

PDF provide that capability, but editors don't produce it, probably because printing is though OS drivers that don't support it, or PDF generators that don't support it. Or they do support it but users don't know to check that option, or turn it off because it makes PDFs too large.

[user\_7832](https://news.ycombinator.com/user?id=user_7832) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958861)

Do you know what this field/type is called, and I’d any of the big names (MS/Adobe etc) support creating such PDFs?

[bux93](https://news.ycombinator.com/user?id=bux93) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959942)

OCR software like ABBY can spit out something called a "searchable PDF", which has a text layer underneath a picture of a scan. Otherwise, PDF has 'dictionaries' with arbitrary key-value pairs in them. The "Info" dictionary has some specific metadata fields like Author, and a "Font" dictionary embeds fonts, but you're free to use those dictionaries for whatever. There's also a standard to embed 'dublin core', rights management and custom metadata called XMP. Files can be embedded. You can also use comments, as PDF is a subset of postscript. When a PDF gets converted to PDF/A (by archiving software) or flattened/optimized, most of these are likely to be lost.

[dkjaudyeqooe](https://news.ycombinator.com/user?id=dkjaudyeqooe) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960671)

I believe it's a "hybrid PDF" but I'm not sure if there's a further standard for merely embedding text.

[https://stackoverflow.com/questions/67358370/what-the-standa...](https://stackoverflow.com/questions/67358370/what-the-standard-used-by-a-hybrid-pdf-file)

[groby\_b](https://news.ycombinator.com/user?id=groby_b) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957655)

PDF supports that just fine. It's just that many PDF publishers choose not to use that.

You can lead a horse to water...

[shermantanktop](https://news.ycombinator.com/user?id=shermantanktop) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42963880)

PDFs began as just postscript commands stored in a file. It’s a genius hack in a way that has become a Frankenstein’s monster.

[nitwit005](https://news.ycombinator.com/user?id=nitwit005) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960357)

People kind of dump whatever in pdf files, so I don't think a cleaner file format would do as much as you might think.

Digital fax services will generate pdf files, for example. They're just image data dumped into a pdf. Various scanners will also do so.

[shermantanktop](https://news.ycombinator.com/user?id=shermantanktop) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956872)

is "put this glyph at coordinate (x,y)" really what you'd call "structured"?

[dkjaudyeqooe](https://news.ycombinator.com/user?id=dkjaudyeqooe) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957640)

He's calling PDFs unstructured: structured editors -> unstructured PDF -> structured data

[irjustin](https://news.ycombinator.com/user?id=irjustin) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959081)

It's not the structure that allows meaningful understanding.

Something that was clearly a table now becomes a bunch of glphy's physically close to eachother vs a group of other glphys but when considered as a group is a box visually separated from another group of glphys but actually part of a table.

[surfingdino](https://news.ycombinator.com/user?id=surfingdino) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961573)

In my experience AWS Textextract does a pretty good job without using LLMs.

[Bluestein](https://news.ycombinator.com/user?id=Bluestein) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42967188)

... *and* call's it "portable", to boot.-

[silverliver](https://news.ycombinator.com/user?id=silverliver) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959542)

We are driving full speed into a xerox 2.0 moment and this time we are doing so knowingly. At least with xerox, the errors were out of place and easy to detect by a human. I wonder how many innocent people will lose their lives or be falsely incarcerated because of this. I wonder if we will adapt our systems and procedures to account for hallucinations and "85%" accuracy.

And no, outlawing use the use of AI or increasing liability with its use will have next to nothing to deter its misuse and everyone knows it. My heart goes out to the remaining 15%.

[anon373839](https://news.ycombinator.com/user?id=anon373839) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960051)

I love generative AI as a technology. But the worst thing about its arrival has been the reckless abandonment of all engineering discipline and common sense. It’s embarrassing.

[sschueller](https://news.ycombinator.com/user?id=sschueller) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960205)

CCC talk about Xerox copiers changing numbers when doing OCR:

[https://media.ccc.de/v/31c3\_-\_6558\_-\_de\_-\_saal\_g\_-\_201412282...](https://media.ccc.de/v/31c3_-_6558_-_de_-_saal_g_-_201412282300_-_traue_keinem_scan_den_du_nicht_selbst_gefalscht_hast_-_david_kriesel)

[tomrod](https://news.ycombinator.com/user?id=tomrod) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42963173)

Would be nice to get a translation for a broader audience, glad folks are reporting this out!

[sschueller](https://news.ycombinator.com/user?id=sschueller) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42963496)

There is a translated one: [https://www.youtube.com/watch?v\=zXXmhxbQ-hk](https://www.youtube.com/watch?v=zXXmhxbQ-hk)

[ikrenji](https://news.ycombinator.com/user?id=ikrenji) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42965231)

the first thing that guy says that existing non-AI solutions are not that great. then he says that AI beats them in the accuracy. so i don't quite understand the point you're trying to make here

[csomar](https://news.ycombinator.com/user?id=csomar) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960613)

Humans accept a degree of error for convenience. (driving is one of them). But no, 15% is not the acceptable rate. More like 0.15% to 0.015% depending on the country.

[tomrod](https://news.ycombinator.com/user?id=tomrod) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42963163)

Meh, just maintain an audit log and an escalation subsystem. No need to be luddites when the problems are process, not tech stack.

[freezed8](https://news.ycombinator.com/user?id=freezed8) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42957085)

(disclaimer I am CEO of llamaindex, which includes LlamaParse)

Nice article! We're actively benchmarking Gemini 2.0 right now and if the results are as good as implied by this article, heck we'll adapt and improve upon it. Our goal (and in fact the reason our parser works so well) is to always use and stay on top of the latest SOTA models and tech :) - we blend LLM/VLM tech with best-in-class heuristic techniques.

Some quick notes: 1. I'm glad that LlamaParse is mentioned in the article, but it's not mentioned in the performance benchmarks. I'm pretty confident that our most accurate modes are at the top of the table benchmark - our stuff is pretty good.

\2. There's a *long* tail of issues beyond just tables - this includes fonts, headers/footers, ability to recognize charts/images/form fields, and as other posters said, the ability to have fine-grained bounding boxes on the source elements. We've optimized our parser to tackle all of these modes, and we need proper benchmarks for that.

\3. DIY'ing your own pipeline to run a VLM at scale to parse docs is surprisingly challenging. You need to orchestrate a robust system that can screenshot a bunch of pages at the right resolution (which can be quite slow), tune the prompts, and make sure you're obeying rate limits \+ can retry on failure.

[M4v3R](https://news.ycombinator.com/user?id=M4v3R) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961205)

The very first (and probably hand-picked & checked) example on your website \[0\] suffers from the very problem people are talking about here - in "Fiscal 2024" row it contains an error for CEO CAP column. On the image it says "$234.1" but the parsed result says "$234.4". A small error, but error nonetheless. I wonder if we can ever fix these kind of errors with LLM parsing.

\[0\] [https://www.llamaindex.ai/llamaparse](https://www.llamaindex.ai/llamaparse)

[dilDDoS](https://news.ycombinator.com/user?id=dilDDoS) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42966827)

Looks like this was fixed, the parsed result says "$234.1" on my end. I wonder if the error was fixed manually or with another round of LLM parsing?

[heidarb](https://news.ycombinator.com/user?id=heidarb) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961499)

I'm a happy customer. I wrote a ruby client for your API and have been parsing thousands of different types of PDFs through it with great results. I tested almost everything out there at the time and I couldn't find anything that came close to being as good as llamaparse.

[BenGosub](https://news.ycombinator.com/user?id=BenGosub) [on Feb 7, 2025](https://news.ycombinator.com/item?id=42971030)

Indeed, this is also my experience. I have tried a lot of things and where quality is more important than quantity, I doubt there are many tools that can come close to Llamaparse.

[rendaw](https://news.ycombinator.com/user?id=rendaw) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960910)

All your examples are exquisitely clean digital renders of digital documents. How does it fare with real scans (noise, folds) or photos? Receipts?

Or is there a use case for digital non-text pdfs? Are people really generating image and not text-based PDFs? Or is the primary use case extracting structure, rather than text?

[rahimnathwani](https://news.ycombinator.com/user?id=rahimnathwani) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957169)

Hi Jerry,

How well does llamaparse work on foreign-language documents?

I have pipeline for Arabic-language docs using Azure for OCR and GPT-4o-mini to extract structured information. Would it be worth trying llamaparse to replace part of the pipeline or the whole thing?

[freezed8](https://news.ycombinator.com/user?id=freezed8) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957341)

yes! we have foreign language support for better OCR on scans. Here's some more details. Docs: [https://docs.cloud.llamaindex.ai/llamaparse/features/parsing...](https://docs.cloud.llamaindex.ai/llamaparse/features/parsing_options) Notebook: [https://github.com/run-llama/llama\_parse/blob/main/examples/...](https://github.com/run-llama/llama_parse/blob/main/examples/demo_languages.ipynb)

[rahimnathwani](https://news.ycombinator.com/user?id=rahimnathwani) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957364)

What is disable\_ocr\=True for? Is it for documents that already have a text layer, that you don't want to OCR again?

[freezed8](https://news.ycombinator.com/user?id=freezed8) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957577)

yeah disable OCR is for documents where you don't need to OCR a scanned image, it'll just parse out the text

it's faster if True

[sensanaty](https://news.ycombinator.com/user?id=sensanaty) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42962147)

There's an error right on your landing page \[1\] with the parsed document...

It's supposed to say 234.1, not 234.4

[https://www.llamaindex.ai/llamaparse](https://www.llamaindex.ai/llamaparse)

[7thpower](https://news.ycombinator.com/user?id=7thpower) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961979)

But can it do this table?!:

[https://x.com/preston\_mos/status/1853931388929511619?s\=46](https://x.com/preston_mos/status/1853931388929511619?s=46)

[rjurney](https://news.ycombinator.com/user?id=rjurney) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954025)

I've been using NotebookLM powered by Gemini 2.0 for three projects and it is \_really powerful\_ for comprehending large corpuses you can't possibly read and thinking informed by all your sources. It has solid Q&A. When you ask a question or get a summary you like \[which often happens\] you can save it as a new note, putting it into the corpus for analysis. In this way your conclusions snowball. Yes, this experience actually happens and it is beautiful.

I've tried Adobe Acrobat AI for this and it doesn't work yet. NotebookLM is it. The grounding is the reason it works - you can easily click on anything and it will take you to the source to verify it. My only gripe is that the visual display of the source material is \_dogshit ugly\_, like exceptionally so. Big blog pink background letters in lines of 24 characters! :) It has trouble displaying PDF columns, but at least it parses them. The ugly will change I'm sure :)

My projects are setup to let me bridge the gaps between the various sources and synthesize something more. It helps to have a goal and organize your sources around that. If you aren't focused, it gets confused. You lay the groundwork in sources and it helps you reason. It works so well I feel \_tender\_ towards it :) Survey papers provide background then you add specific sources in your area of focus. You can write a profile for how you would like NotebookLM to think - which REALLY helps out.

They are:

\* The Stratigrapher - A Lovecraftian short story about the world's first city. All of Seton Lloyd/Faud Safar's work on Eridu. Various sources on Sumerian culture and religion All of Lovecraft's work and letters. Various sources about opium Some articles about nonlinear geometries

\* FPGA Accelerated Graph Analytics An introduction to Verilog Papers on FPGAs and graph analytics Papers on Apache Spark architecture Papers on GraphFrames and a related rant I created about it and graph DBs A source on Spark-RAPIDS Papers on subgraph matching, graphlets, network motifs Papers on random graph models

\* Graph machine learning notebook without a specific goal, which has been less successful. It helps to have a goal for the project. It got confused by how broad my sources were.

I would LOVE to share my projects with you all, but you can only share within a Google Workspaces domain. It will be AWESOME when they open this thing up :)

[anirudhb99](https://news.ycombinator.com/user?id=anirudhb99) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954189)

thanks a ton for all the amazing feedback on this thread! if

(a) you have document understanding use cases that you'd like to use gemini for (the more aspirational the better) and/or

(b) there are loss cases for which gemini doesn't work well today,

please feel free to email anirudhbaddepu@google.com and we'd love to help get your use case working & improve quality for our next series of model updates!

[nabla9](https://news.ycombinator.com/user?id=nabla9) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960592)

What if you need scan pages from thick paper books or binded documents without specialized book scanner?

I have two user cases in mind:

\1. Photographs of open book.

\2. Having video feed of open book where someone flips pages manually.

[rudolph9](https://news.ycombinator.com/user?id=rudolph9) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955385)

We parse millions of PDFs using Apache Tika and process about 30,000 per dollar of compute cost. However, the structured output leaves something to be desired, and there are a significant number of pages that Tika is unable to parse.

[https://tika.apache.org/](https://tika.apache.org/)

[rudolph9](https://news.ycombinator.com/user?id=rudolph9) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956361)

Under the hood tika uses tesseract for ocr parsing. For clarity this all works surprisingly well generally speaking and it’s pretty easy to run your self and order of magnitude cheaper than most services out there.

[https://tesseract-ocr.github.io/tessdoc/](https://tesseract-ocr.github.io/tessdoc/)

[gapeslape](https://news.ycombinator.com/user?id=gapeslape) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956276)

In my mind, Gemini 2.0 changes everything because of the incredibly long context (2M tokens on some models), while having strong reasoning capabilities.

We are working on compliance solution ([https://fx-lex.com](https://fx-lex.com)) and RAG just doesn’t cut it for our use case. Legislation cannot be chunked if you want the model to reason well about it.

It’s magical to be able to just throw everything into the model. And the best thing is that we automatically benefit from future model improvements along all performance axes.

[pvo50555](https://news.ycombinator.com/user?id=pvo50555) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959708)

What does "throw everything into the model" entail in your context?

How much data are you able to feed into the model in a single prompt and on what hardware, if I may ask?

[gapeslape](https://news.ycombinator.com/user?id=gapeslape) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960503)

Gemini models run in the cloud, so there is no issue with hardware.

The EU regulations typically include delegated acts, technical standards, implementation standards and guidelines. With Gemini 2.0 we are able to just throw all of this into the model and have it figure out.

This approach gives way better results than anything we are able to achieve with RAG.

My personal bet is that this is how the future will look like. RAG will remain relevant, but only for extremely large document corpuses.

[manmal](https://news.ycombinator.com/user?id=manmal) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957222)

Maybe a dumb question, have you tried fine tuning on the corpus, and then adding a reasoning process (like all those R1 distillations)?

[gapeslape](https://news.ycombinator.com/user?id=gapeslape) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960569)

We haven't tried that, we might do that in the future.

My intuition - not based on any research - is that recall should be a lot better from in context data vs. weights in the model. For our use case, precise recall is paramount.

[galvin](https://news.ycombinator.com/user?id=galvin) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958190)

Somewhat tangential, but the EU has a directive mandating electronic invoicing for public procurement.

One of the standards that has come out of that is EN 16931, also known as ZUGFeRD and Factur-X, which basically involves embedding an XML file with the invoice details inside a PDF/A. It allows the PDF to be used like a regular PDF but it also allows the government procurement platforms to reliably parse the contents without any kind of intelligence.

It seems like a nice solution that would solve a lot of issues with ingesting PDFs for accounting if everyone somehow managed to agree a standard. Maybe if EN 16931 becomes more broadly available it might start getting used in the private sector too.

[jbarrow](https://news.ycombinator.com/user?id=jbarrow) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954299)

\> Unfortunately Gemini really seems to struggle on this, and no matter how we tried prompting it, it would generate wildly inaccurate bounding boxes

Qwen2.5 VL was trained on a special HTML format for doing OCR with bounding boxes. \[1\] The resulting boxes aren't quite as accurate as something like Textract/Surya, but I've found they're much more accurate than Gemini or any other LLM.

\[1\] [https://qwenlm.github.io/blog/qwen2.5-vl/](https://qwenlm.github.io/blog/qwen2.5-vl/)

[fngjdflmdflg](https://news.ycombinator.com/user?id=fngjdflmdflg) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953568)

\>Unfortunately Gemini really seems to struggle on this, and no matter how we tried prompting it, it would generate wildly inaccurate bounding boxes

This is what I have found as well. From what I've read, LLMS do not work well with images for specific details due to image encoders which are too lossy. (No idea if this is actually correct.) For now I guess you can use regular OCR to get bounding boxes.

[minimaxir](https://news.ycombinator.com/user?id=minimaxir) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953670)

Modern multimodal encoders for LLMs are fine/not lossy since they do not resize to a small size and can handle arbitrary sizes, although some sizes are obviously better represented in the training set. A 8.5" x 11" paper would be common.

I suspect the issue is prompt engineering related.

\> Please provide me strict bounding boxes that encompasses the following text in the attached image? I'm trying to draw a rectangle around the text.

\> - Use the top-left coordinate system

\> - Values should be percentages of the image width and height (0 to 1)

LLMs have enough trouble with integers (since token-wise integers and text representation of integers are the same), high-precision decimals will be even worse. It might be better to reframe the problem as "this input document is 850 px x 1100 px, return the bounding boxes as integers" then parse and calculate the decimals later.

[fngjdflmdflg](https://news.ycombinator.com/user?id=fngjdflmdflg) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953889)

Just tried this and it did not appear to work for me. Prompt:

\>Please provide me strict bounding boxes that encompasses the following text in the attached image? I'm trying to draw a rectangle around the text.

\> - Use the top-left coordinate system

\>this input document is 1080 x 1236 px. return the bounding boxes as integers

[BoorishBears](https://news.ycombinator.com/user?id=BoorishBears) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955323)

[https://github.com/google-gemini/cookbook/blob/a916686f95f43...](https://github.com/google-gemini/cookbook/blob/a916686f95f43aaef200875ac7174082dbdc4e76/quickstarts/Spatial_understanding.ipynb)

They say there's no magic prompt but I'd start with their default since there is usually *some* format used to improve performance with posttraining with tasks like this

[minimaxir](https://news.ycombinator.com/user?id=minimaxir) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953917)

"Might" being the operative word, particularly with models that have less prompt adherence. There's a few other prompt massaging tricks beyond the scope of a HN comment, the decimal issue is just one optimization.

[kbyatnal](https://news.ycombinator.com/user?id=kbyatnal) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955236)

It's clear that OCR & document parsing are going to be swallowed up by these multimodal models. The best representation of a document at the end of the day is an image.

I founded a doc processing company \[1\] and in our experience, a lot of the difficulty w/ deploying document processing into production is when accuracy requirements are high (> 97%). This is because OCR and parsing is only one part of the problem, and real world use cases need to bridge the gap between raw outputs and production-ready data.

This requires things like:

\- state-of-the-art parsing powered by VLMs and OCR

\- multi-step extraction powered by semantic chunking, bounding boxes, and citations

\- processing modes for document parsing, classification, extraction, and splitting (e.g. long documents, or multi-document packages)

\- tooling that lets nontechnical members quickly iterate, review results, and improve accuracy

\- evaluation and benchmarking tools

\- fine-tuning pipelines that turn reviewed corrections —> custom models

Very excited to get test and benchmark Gemini 2.0 in our product, very excited about the progress here.

\[1\] [https://extend.app/](https://extend.app/)

[anon373839](https://news.ycombinator.com/user?id=anon373839) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955931)

\> It's clear that OCR & document parsing are going to be swallowed up by these multimodal models.

I don’t think this is clear at all. A multimodal LLM can and will hallucinate data at arbitrary scale (phrases, sentences, etc.). Since OCR is the part of the system that extracts the “ground truth” out of your source documents, this is an unacceptable risk IMO.

[ALittleLight](https://news.ycombinator.com/user?id=ALittleLight) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956985)

Seems like you could solve hallucinations by repeating the task multiple times. Non-hallucinations will be the same. Hallucinations will be different. Discard and retry hallucinated sections. This increases cost by a fixed multiple, but if cost of tokens continues to fall that's probably perfectly fine.

[nnurmanov](https://news.ycombinator.com/user?id=nnurmanov) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959372)

If you see above, someone is using a second and even third LLM to correct LLM outputs, I think it is the way to minimize hallucinations.

[otabdeveloper4](https://news.ycombinator.com/user?id=otabdeveloper4) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960032)

\> I think it is the way to minimize hallucinations

Or maybe the way to add new hallucinations. Nobody really knows. Just trust us bro, this is groundbreaking disruptive technology.

[esjeon](https://news.ycombinator.com/user?id=esjeon) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959543)

I think professional services will continue to use OCRs in one way or another, because it's simply too cheap, fast, and accurate. Perhaps, multi-modal models can help address shortcomings of OCRs, like layout detection and guessing unrecognizable characters.

[\_\_jl\_\_](https://news.ycombinator.com/user?id=__jl__) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954147)

The numbers in the blog post seem VERY inaccurate.

Quick calculation: Input pricing: Image input in 2.0 Flash is $0.0001935. Let's ignore the prompt. Output pricing: Let's assume 500 token per page, which is $0.0003

Cost per page: $0.0004935

That means 2,026 pages per dollar. Not 6,000!

Might still be cheaper than many solutions but I don't see where these numbers are coming from.

By the way, image input is much more expensive in Gemini 2.0 even for 2.0 Flash Lite.

Edit: The post says batch pricing, which would be 4k pages based on my calculation. Using batch pricing is pretty different though. Great if feasible but not practical in many contexts.

[serjester](https://news.ycombinator.com/user?id=serjester) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954641)

Correct, it's with batching Vertex pricing with slightly lower output tokens per page since a lot of pages are somewhat empty in real world docs - I wanted a fair comparison to providers that charge per page.

Regardless of what assumptions you use - it's still an order of magnitude \+ improvement over anything else.

[GGByron](https://news.ycombinator.com/user?id=GGByron) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958503)

I've not followed the literature very closely for some time - what problem are they trying to solve in the first place? They write "for documents to be effectively used in RAG pipelines, they must be split into smaller, semantically meaningful chunks". Segmenting each page by paragraphs doesn't seem like a particularly hard vision problem, nor do I see why an OCR system would need to incorporate an LLM (which seem more like a demonstration of overfitting than a "language model" in any literal sense, going by ChatGPT). Perhaps I'm just out of the loop.

Finally, I must point out that statements in the vein of "Why \[product\] 2.0 Changes Everything" are more often than not a load of humbug.

[beklein](https://news.ycombinator.com/user?id=beklein) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953884)

Great article, I couldn't find any details about the prompt... only the snippets of the \`CHUNKING\_PROMPT\` and the \`GET\_NODE\_BOUNDING\_BOXES\_PROMPT\`.

Is there is any code example with a full prompt available from OP, or are there any references (such as similar GitHub repos) for those looking to get started within this topic?

Your insights would be highly appreciated.

[pmarreck](https://news.ycombinator.com/user?id=pmarreck) [on Feb 17, 2025](https://news.ycombinator.com/item?id=43081004)

I have some out-of-print books that I want to convert into nice pdf's/epubs (like, reference-quality)

\1) I don't mind destroying the binding to get the best quality. How do I do so?

\2) I have a multipage double-sided scanner (fujitsu scansnap). would this be sufficient to do the scan portion?

\3) Is there anything that determines the font of the book text and reproduces that somehow? and that deals with things like bold and italic and applies that either as markdown output or what have you?

\4) how do you de-paginate the raw text to reflow into (say) an epub format that will paginate based on the output device specification?

[roywashere](https://news.ycombinator.com/user?id=roywashere) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955768)

I think it is very ironic that we chose to use PDF in many fields to archive data because it is a standard and because we would be able to open our pdf documents in 50 or 100 years time. So here we are just a couple of years later facing the challenge of getting the data out of our stupid PDF documents already!

[esafak](https://news.ycombinator.com/user?id=esafak) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956228)

It's not ironic. PDFs are a *container*, which can hold scanned documents as well as text. Scanned documents need OCR and to be analyzed for their layout. This is not a failing of the PDF format, but a problem inherent to working with print scans.

I don't claim PDF is a good format. It is inscrutable to me.

[scotty79](https://news.ycombinator.com/user?id=scotty79) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956533)

Pdf is a horrible format. Even if it contains plain text it has no concept of something as simple as paragraphs.

One can wonder how much wonkiness of llms comes from errors in extracting language from pdfs.

Adobe is the most harmful software development company in existence.

[phoh](https://news.ycombinator.com/user?id=phoh) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957722)

amen

[ChrisArchitect](https://news.ycombinator.com/user?id=ChrisArchitect) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953733)

Related:

*Gemini 2.0 is now available to everyone*

[https://news.ycombinator.com/item?id\=42950454](https://news.ycombinator.com/item?id=42950454)

[rjcrystal](https://news.ycombinator.com/user?id=rjcrystal) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958773)

I work in healthcare domain, We've had great success converting printed lab reports (95%) to Json format using 1.5-Flash model. This post is really exciting for me. will definitely try out 2.0 models.

The struggle which almost every ocr usecase faces is with handwritten documents(doctor prescriptions with bad handwriting) With gemini 1.5 flash we've had \~75-80% percent accuracy (based on random sampling by pharmacists). we're planning to improve this further by fine-tuning gemini models with medical data.

What could be other alternative services/models for accurate handwriting ocr?

[myaccountonhn](https://news.ycombinator.com/user?id=myaccountonhn) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960475)

\> We've had great success converting printed lab reports (95%) to Json format using 1.5-Flash model

Sounds terrifying. How can you be sure that there were no conversion mistakes?

[Havoc](https://news.ycombinator.com/user?id=Havoc) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960899)

How on earth is anyone ok with 75% accuracy in prescriptions context?!? Or medical anything

That’s literally insane

[martimchaves](https://news.ycombinator.com/user?id=martimchaves) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961428)

I'm guessing that human accuracy may be lower or around that value, given that handwritten notes are generally difficult to read. A better metric for document parsing might be accuracy relative to human performance (how much better the LLM performs compared to a human).

[knowaveragejoe](https://news.ycombinator.com/user?id=knowaveragejoe) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42966692)

Nobody said they're okay with it, nor did they describe what they use the data for.

[erulabs](https://news.ycombinator.com/user?id=erulabs) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956396)

Hrm I've been using a combo of Textract (for bounding boxes) AI for understanding the contents of the document. Textract is excellent at bounding boxes and exact-text capture, but LLMs are excellent at understanding when a messy/ugly bit of a form is actually one question, or if there are duplicate questions etc.

Correlating the two (Textract \<-> AI) output is difficult, but another round of AI is usually good at that. Combined with some text-different scoring and logic, I can get pretty good full-document understanding of questions and answer locations. I've spent a pretty absurd amount of time on this and as of yet have not launched a product with it, but if anyone is interested I'd love to chat about the pipeline!

[Havoc](https://news.ycombinator.com/user?id=Havoc) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953438)

Been toying with the flash model. Not the top model, but think it'll see plenty use due to the details. Wins on things other than top of benchmark logs

\* Generous free tier

\* Huge context window

\* Lite version feels basically instant

However

\* Lite model seems more prone to repeating itself / looping

\* Very confusing naming e.g. {model}-latest worked for 1.5 but now its {model}-001? The lite has a date appended, the non-lite does not. Then there is exp and thinking exp...which has a date. wut?

[ai-christianson](https://news.ycombinator.com/user?id=ai-christianson) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953462)

\> \* Huge context window

But how well does it actually handle that context window? E.g. a lot of models support 200K context, but the LLM can only really work with \~80K or so of it before it starts to get confused.

[asadm](https://news.ycombinator.com/user?id=asadm) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953554)

it works REALLY well. I have used it to dump many references codes and then help me write a new modules etc. I have gone up to 200k tokens I think with no problems in recall.

[ai-christianson](https://news.ycombinator.com/user?id=ai-christianson) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953640)

Awesome. Models that can usefully leverage such large context windows are rare at this point.

Something like this opens up a lot of use cases.

[Havoc](https://news.ycombinator.com/user?id=Havoc) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953536)

I'm sure someone will do a haystack test, but from my casual testing it seems pretty good

[llm\_nerd](https://news.ycombinator.com/user?id=llm_nerd) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955202)

There is the needle in the haystack measure which is, as you probably guessed, hiding a small fact in a massive set of tokens and asking it to recall it.

Recent Gemini models actually do extraordinarily well.

[https://cloud.google.com/blog/products/ai-machine-learning/t...](https://cloud.google.com/blog/products/ai-machine-learning/the-needle-in-the-haystack-test-and-how-gemini-pro-solves-it)

[f38zf5vdt](https://news.ycombinator.com/user?id=f38zf5vdt) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953762)

It works okay out to roughly 20-40k tokens. Once the window gets larger than that, it degrades significantly. You can needle in the haystack out to that distance, but asking it for multiple things from the document leads to hallucinations for me.

Ironic, but GPT4o works better for me at longer contexts \<128k than Gemini 2.0 flash. And out to 1m is just hopeless, even though you can do it.

[summerlight](https://news.ycombinator.com/user?id=summerlight) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953514)

My experience is that Gemini works relatively well on larger contexts. Not perfect, but more reliable.

[anonu](https://news.ycombinator.com/user?id=anonu) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42964875)

Ingesting PDFs accurately is a noble goal which will no doubt be solved as LLMs get better. However, I need to point out that the financial statement example used in the article already has a solution: iXBRL.

Many financial regulators require you to publish heavily marked up statements with iXBRL. These markups reveal nuances in the numbers that OCRing a post processed table will not understand.

Of course, financial documents are a narrow subset of the problem.

Maybe the problem is with PDF as a format: Unfortunately PDFs lose that meta information when they are built from source documents.

I can't help but feel that PDFs could probably be more portable as their acronym indicates.

[tomrod](https://news.ycombinator.com/user?id=tomrod) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42964996)

Just call out -- even better, this library (even in active development) is blowing every other SEC tool I've found out the of the water

[https://github.com/dgunning/edgartools](https://github.com/dgunning/edgartools)

[xnx](https://news.ycombinator.com/user?id=xnx) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955622)

Glad Gemini is getting some attention. Using it is like a superpower. There are so many discussions about ChatGTP, Claude, DeepSeek, Llama, etc. that don't even mention Gemini.

[Workaccount2](https://news.ycombinator.com/user?id=Workaccount2) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956190)

Before 2.0 models their offerings were pretty underwhelming, but now they can certainly hold their own. I think Gemini will ultimately be the LLM that eats the world, Google has the talent and most importantly has their own custom hardware (hence why their prices are dirt cheap and context is huge).

[throwaway314155](https://news.ycombinator.com/user?id=throwaway314155) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955696)

Google had a pretty rough start compared to ChatGPT, Claude. I suspect that left a bad taste in many people's mouths. In particular because evaluating so many LLM's is a lot of effort on its own.

Llama and DeepSeek are no-brainers; the weights are public.

[beastman82](https://news.ycombinator.com/user?id=beastman82) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955879)

No brainer if you're sitting on a >$100k inference server.

[throwaway314155](https://news.ycombinator.com/user?id=throwaway314155) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956391)

Sure, that's fair. If you're aiming for state of the art performance. Otherwise, you can get close and do it on reasonably priced hardware by using smaller distilled and/or quantized variants of llama/r1.

Really though I just meant "it's a no-brainer that they are popular here on HN".

[BoorishBears](https://news.ycombinator.com/user?id=BoorishBears) [on Feb 7, 2025](https://news.ycombinator.com/item?id=42969919)

I pay 78 cents an hour to host Llama.

[beastman82](https://news.ycombinator.com/user?id=beastman82) [on Feb 8, 2025](https://news.ycombinator.com/item?id=42982334)

Vast? Specs?

[BoorishBears](https://news.ycombinator.com/user?id=BoorishBears) [on Feb 9, 2025](https://news.ycombinator.com/item?id=42988133)

Runpod, 2xA40.

Not sure why you think buying an entire inference server is a necessity to run these models.

[sumedh](https://news.ycombinator.com/user?id=sumedh) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955982)

Google was not serious about LLMs, they could not even figure what to call it. There is always a risk that they will get bored and just kill the whole thing.

[tomasello77](https://news.ycombinator.com/user?id=tomasello77) [on Feb 14, 2025](https://news.ycombinator.com/item?id=43047217)

I tried using Gemini 2.0 Flash for PDF-to-Markdown parsing of scientific papers after having good results with GPT-4o, but the experience was terrible.

When I sent images of PDF page with extracted text, Gemini mixed headlines with body text, parsed tables incorrectly, and sometimes split tables—placing one part at the top of the page and the rest at the bottom. It also added random numbers (like inserting an “8” for no reason).

When using the Gemini SDK to process full PDFs, Gemini 1.5 could handle them, but Gemini 2.0 only processed the first page. Worse, both versions completely ignored tables.

Among the Gemini models, 1.5 Pro performed the best, reaching about 80% of GPT-4o’s accuracy with image parsing, but it still introduced numerous small errors.

In conclusion, no Gemini model is reliable for PDF-to-Markdown parsing and beyond the hype - I still need to use GPT-4o.

[oedemis](https://news.ycombinator.com/user?id=oedemis) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955441)

there is also [https://ds4sd.github.io/docling/](https://ds4sd.github.io/docling/) from ibm research which is mit license and track bounding boxes as rich json format

[pbronez](https://news.ycombinator.com/user?id=pbronez) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956458)

Docling has worked well for me. It handles scenarios that crashed ChatGPT Pro. Only problem is it's super annoying to install. When I have a minute I might package it for homebrew.

[disqard](https://news.ycombinator.com/user?id=disqard) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958184)

Did you compare it to tesseract?

If it's superior (esp. for scans with text flowing around image boxes), and if you do end up packaging it up for brew, know that there's at least one developer who will benefit from your work (for a side-project, but that goes without saying).

Thanks in advance!

[nsmurali](https://news.ycombinator.com/user?id=nsmurali) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961894)

I have seen no decent program that can read, OCR, and analyze, and tabulate data correctly from very large PDF files with a lot of scanned information from different sources. I run my practice with pdf files- one for each patient. It is a treasure trove of actionable data. PDF filing in this manner allows me to finish my daily tasks in 4 hrs instead of 12 hrs! For sick patients who need information at the point of care, PDF has numerous advantages over usual hospital EHR portals, etc. If any smart Engineer/s are interested in working with me, please connect with me

[7thpower](https://news.ycombinator.com/user?id=7thpower) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961935)

I can help as can many others. Probably a good place to start though is with some of the more recent off the shelf solutions like trellis (I have no affiliation with them).

[bt3](https://news.ycombinator.com/user?id=bt3) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953466)

One major takeaway that matches my own investigation is that Gemini 2.0 still materially struggles with bounding boxes on digital content. Google has published\[1\] some great material on spatial understanding and bounding boxes on photography, but identifying sections of text or digital graphics like icons in a presentation is still very hit and miss.

--

\[1\]: [https://github.com/google-gemini/cookbook/blob/a916686f95f43...](https://github.com/google-gemini/cookbook/blob/a916686f95f43aaef200875ac7174082dbdc4e76/quickstarts/Spatial_understanding.ipynb)

[maeil](https://news.ycombinator.com/user?id=maeil) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953840)

Have you seen any models that perform better at this? I last looked into this a year ago but at the time they were indeed quite bad at it across the board.

[eviks](https://news.ycombinator.com/user?id=eviks) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958807)

What would change "everything" is if we managed to switch to "real" digital parseable formats instead of this dead tree emulation that buries all data before the arrival of AI...

[scottydelta](https://news.ycombinator.com/user?id=scottydelta) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953493)

This is what I am trying to figure out how to solve.

My problem statement is:

\- Injest PDFs, summarize, and extract important information.

\- Have some way to overlay the extracted information on the pdf in the UI.

\- User can provide feedback on the overlaid info by accepting or rejecting the highlights as useful or not.

\- This info goes back in to the model for reinforced learning.

Hoping to find something that can make this more manageable.

[cccybernetic](https://news.ycombinator.com/user?id=cccybernetic) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953907)

Most PDF parsers give you coordinate data (bounding boxes) for extracted text. Use these to draw highlights over your PDF viewer - users can then click the highlights to verify if the extraction was correct.

The tricky part is maintaining a mapping between your LLM extractions and these coordinates.

One way to do it would be with two LLM passes:

```
1. First pass: Extract all important information from the PDF
  2. Second pass: "Hey LLM, find where each extraction appears in these bounded text chunks"
```

 Not the cheapest approach since you're hitting the API twice, but it's straightforward!

[Jimmc414](https://news.ycombinator.com/user?id=Jimmc414) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954090)

Here's a PR thats not accepted yet for some reason that seems to be having some success with the bounding boxes

[https://github.com/getomni-ai/zerox/pull/44](https://github.com/getomni-ai/zerox/pull/44)

Related to

[https://github.com/getomni-ai/zerox/issues/7](https://github.com/getomni-ai/zerox/issues/7)

[baxtr](https://news.ycombinator.com/user?id=baxtr) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953630)

Have you tried cursor or replit for this?

[memhole](https://news.ycombinator.com/user?id=memhole) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42962385)

I’ve been very reluctant to use closed source LLMs. This might actually convince me to use one. I’ve done so many attempts at pdf parsing over the years. It’s awful to deal with. 2 column format omg. Most don’t realize that pdfs contain instructions for displaying the document and the content is buried in there. It’s just always been a problematic format.

So if it works, I’d be a fool not to use it.

[minimalengineer](https://news.ycombinator.com/user?id=minimalengineer) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958027)

Two years ago, I worked for a company that had its own proprietary AI system for processing PDFs. While the system handled document ingestion, its real value was in extracting and analyzing data to provide various insights. However, one key requirement was rendering documents in HTML with as close to a 1:1 likeness as possible.

At the time, I evaluated multiple SDKs for both OCR and non-OCR PDF conversions, but none matched the accuracy of Adobe Acrobat’s built-in solution. In fact, at one point (don’t laugh), the company resorted to running Adobe Acrobat on a Windows machine with automation tools to handle the conversion. Using Adobe’s cloud service for conversion was not an option due to the proprietary nature of the PDFs. Additionally, its results were inconsistent and often worse compared to the desktop version of Adobe Acrobat!

Given that experience, I see this primarily as an HTML/text conversion challenge. If Gemini 2.0 truly improves upon existing solutions, it would be interesting to see a direct comparison against popular proprietary tools in terms of accuracy.

[diptanu](https://news.ycombinator.com/user?id=diptanu) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960441)

We started with using LLMs for parsing at Tensorlake ([https://docs.tensorlake.ai](https://docs.tensorlake.ai)), tried Qwen, Gemini, OpenAI, pretty much everything under the sun. My thought was we could skip 5-6 years of development IDP companies have done on specialized models by going to LLMs.

On information dense pages, LLMs often hallucinate half of the times, they have trouble understanding empty cells in tables, doesn't understand checkboxes, etc.

We had to invest heavily into building a state of the art layout understanding model and finally a table structure understanding for reliability. LLMs will get there, but there are some ways to go there.

Where they do well is in VQA type use cases, ask a question, very narrowly scoped, they will work much better than OCR\+Layout models, because they are much more generalizable and flexible to use.

[mehulashah](https://news.ycombinator.com/user?id=mehulashah) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42965941)

(Disclosure, CEO of Aryn ([https://aryn.ai/](https://aryn.ai/)) here)

Good post. VLM models are improving and Gemini 2.0 definitely changes the doc prep and ingestion pipeline across the board.

What we're finding as we work with enterprise customers:

\1. Attribution is super important, and VLMs are there yet. Combining them with layout analysis makes for a winning combo.

\2. VLMs are great at prompt-based extraction, but if you have document automation and you don't know where in tables you'll be searching or need to reproduce faithfully -- then precise table extraction is important.

\3. VLMs will continue to get better, but the price points are a result of economies of scale that document parsing vendors don't get. On the flip side, document parsing vendors have deployment models that Gemini can't reach.

[cccybernetic](https://news.ycombinator.com/user?id=cccybernetic) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953774)

Shameless plug: I'm working on a startup in this space.

But the bounding box problem hits close to home. We've found Unstructured's API gives pretty accurate box coordinates, and with some tweaks you can make them even better. The tricky part is implementing those tweaks without burning a hole in your wallet.

[xiaofei\_](https://news.ycombinator.com/user?id=xiaofei_) [on Feb 17, 2025](https://news.ycombinator.com/item?id=43081688)

How is their API priced? I checked a few months ago and remembered it being expensive.

[amai](https://news.ycombinator.com/user?id=amai) [on Feb 7, 2025](https://news.ycombinator.com/item?id=42977319)

Better have a look at

\- [https://mathpix.com/](https://mathpix.com/)

\- Docling : [https://ds4sd.github.io/docling/](https://ds4sd.github.io/docling/)

[ThinkBeat](https://news.ycombinator.com/user?id=ThinkBeat) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955283)

Hmm I have been doing a but if this manually lately for a personal project. I am working on some old books that are far past any copyright, but they are not available anywhere on the net. (Being in Norwegian m makes a book a lot more obscure) so I have been working on creating ebooks out of them.

I have a scanner, and some OCR processes I run things through. I am close to 85% from my automatic process.

The pain of going from 85% to 99% though is considerable. (and in my case manual) (well Perl helps)

I went to try this AI on one of the short poem manufscript I have.

I told the prompt I wanted PDF to Markdown, it says sure go ahead give me the pdf. I went upload it. It spent a long time spinning. then a quick messages comes up, something like

"Failed to count tokens"

but it just flashes and goes away.

I guess the PDF is too big? Weird though, its not a lot of pages.

[stigz](https://news.ycombinator.com/user?id=stigz) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958103)

I experienced something similar. My use case is I need to summarize bank statements (sums, averages, etc.). Gemini wouldn't do it, it said too many pages. When I asked the max number of supported pages, it says max is 14 pages. Attempted on both 2.0 flash and 2.0 pro in VertexAI console.

[shijithpk](https://news.ycombinator.com/user?id=shijithpk) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961647)

Try with [https://aistudio.google.com](https://aistudio.google.com) Think the page limit is a vertex thing The only limit in reality is the number of input tokens taken to parse the pdf. If those tokens \+ tokens for the rest of your prompt are under the context window limit, you're good.

[sumedh](https://news.ycombinator.com/user?id=sumedh) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956006)

Take a screenshot of the pdf page and give that to the LLM and see if it can be processed.

Your PDF might have some quirks inside which the LLM cannot process.

[pbronez](https://news.ycombinator.com/user?id=pbronez) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956445)

Wonder how this compares to Docling. So far that's been the only tool that really unlocked PDFs for me. It's solid but really annoying to install.

[https://ds4sd.github.io/docling/](https://ds4sd.github.io/docling/)

[matthest](https://news.ycombinator.com/user?id=matthest) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953990)

This is completely tangential, but does anyone know if AI is creating any new jobs?

Thinking of the OCR vendors who get replaced. Where might they go?

One thing I can think of is that AI could help the space industry take off. But wondering if there are any concrete examples of new jobs being created.

[visarga](https://news.ycombinator.com/user?id=visarga) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959499)

\> Thinking of the OCR vendors who get replaced. Where might they go?

We are solving more complicated document types, in more languages, longer in size. The scope of work expanded a lot.

[mansourdaman](https://news.ycombinator.com/user?id=mansourdaman) [on Feb 9, 2025](https://news.ycombinator.com/item?id=42990210)

I've built a simple OCR tool with gemini 2 flash with several options: 1-Simple OCR: Extracts all detected text from uploaded files 2-Advanced OCR: Enables rule-based extraction (e.g., table data) 3-Bulk OCR: Designed for processing multiple files at once The project will be open-source next week. You can try the tool here: [https://gemini2flashocr.netlify.app](https://gemini2flashocr.netlify.app)

[nickandbro](https://news.ycombinator.com/user?id=nickandbro) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953976)

I think very soon a new model will destroy whatever startups and services are built around document ingestion. As in a model that can take in a pdf page as a image and transcribe it to text with near perfect accuracy.

[layer8](https://news.ycombinator.com/user?id=layer8) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955074)

Extracting plain text isn’t that much of a problem, relatively speaking. It’s interpreting more complex elements like nested lists, tables, side bars, footnotes/endnotes, cross-references, images and diagrams where things get challenging.

[visarga](https://news.ycombinator.com/user?id=visarga) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959407)

OCR is not 100% either. Reading order is also fragile, it might OCR the word but mess up the line structure.

[depr](https://news.ycombinator.com/user?id=depr) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954513)

I think the Azure Document Intelligence, Google Document AI and Amazon Textract are among the best if not the best services though and they offer these models.

[nnurmanov](https://news.ycombinator.com/user?id=nnurmanov) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959514)

I have not tested Azure Document Intelligence, Google Document AI, but AWS Textract, LLamaparse, Unstructured and Omni made to my shortlist. I have not tested Docling, as I could not install it on my Windows laptop.

[BenGosub](https://news.ycombinator.com/user?id=BenGosub) [on Feb 7, 2025](https://news.ycombinator.com/item?id=42970750)

They do not test Llamaparse on the accuracy benchmark. In my personal experience Llamaparse was one of the rare tools that always got the right information. Also, the accuracy is only based on tables and we had issues with irregular text structures as well. It is also worth noting that when using an LLM, a non-deterministic tool to do something deterministic is a bit risky and you need to write, modify and maintain a prompt.

[uri\_merhav](https://news.ycombinator.com/user?id=uri_merhav) [on Feb 7, 2025](https://news.ycombinator.com/item?id=42976697)

Gemini Flash 2.0 is impressive but it hardly captures all of the information in the PDF. It's great for getting vibes from the document or finding overall information in it. If you ask it to e.g. enumerate every line item from multiple tables in a long PDF it still falls flat (dropping some line items or entire sections etc). DocuPanda and to a lesser extent Unstrucutred handle this.

[dwheeler](https://news.ycombinator.com/user?id=dwheeler) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957326)

I wish more PDFs were generated as *hybrid* PDFs. These are PDFs that *also* include their original source material. Then you have a document whose format is fixed, but if you need more semantic information, there it is!

LibreOffice makes this especially easy to do: [https://wiki.documentfoundation.org/Faq/Writer/PDF\_Hybrid](https://wiki.documentfoundation.org/Faq/Writer/PDF_Hybrid)

[daemonologist](https://news.ycombinator.com/user?id=daemonologist) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953440)

I wonder how this compares to open source models (which might be less accurate but even cheaper if self-hosted?), e.g. Llama 3.2. I'll see if I can run the benchmark.

Also regarding the failure case in the footnote, I think Gemini actually got that right (or at least outperformed Reducto) - the original document seems to have what I call a "3D" table where the third axis is rows *within* each cell, and having multiple headers is probably the best approximation in Markdown.

[mediaman](https://news.ycombinator.com/user?id=mediaman) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953492)

Everything I tried previously had very disappointing results. I was trying to get rid of Azure's DocumentIntelligence, which is kind of expensive at scale. The models could often output a portion of a table, but it was nearly impossible to get them to produce a structured output of a large table on a single page; they'd often insert "...rest of table follows" and similar terminations, regardless of different kinds of prompting.

Maybe incremental processing of chunks of the table would have worked, with subsequent stitching, but if Gemini can just process it that would be pretty good.

[rp36](https://news.ycombinator.com/user?id=rp36) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958904)

I'm failing to understanding the Ingesting part of the Gemini 2.0? Does Gemini provide the a process to convert PDFs to Markdown API OR the LLM APIs handle it with prompt "Extract the Attached PDF" using this API: [https://ai.google.dev/gemini-api/docs/document-processing?la...](https://ai.google.dev/gemini-api/docs/document-processing?lang=node)

[zoogeny](https://news.ycombinator.com/user?id=zoogeny) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954660)

Orthogonal to this post, but this just highlights the need for a more machine readable PDF alternative.

I get the inertia of the whole world being on PDF. And perhaps we can just eat the cost and let LLMs suffer the burden going forwards. But why not use that LLM coding brain power to create a better overall format?

I mean, do we really see printing things out onto paper something we need to worry about for the next 100 years? It reminds me of the TTY interface at the heart of Linux. There was a time it all made sense, but can we just deprecate it all now?

[layer8](https://news.ycombinator.com/user?id=layer8) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955191)

PDF does support incorporating information about the logical document structure, aka Tagged PDF. It’s optional, but recommended for accessibility (e.g. PDF/UA). See chapters 14.7–14.8 in \[1\]. Processing PDF files as rendered images, as suggested elsewhere in this thread, can actually dramatically lose information present in the PDF.

Alternatively, XML document formats and the like do exist. Indeed, HTML was supposed to be a document format. That’s not the problem. The problem is having people and systems actually author documents in that way in an unambiguous fashion, and having a uniform visual presentation for it that would be durable in the long term (decades at least).

PDF as a format persists because it supports virtually every feature under the sun (if authors care to use them), while largely guaranteeing a precisely defined visual presentation, and being one of the most stable formats.

\[1\] [https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandard](https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandard)...

[zoogeny](https://news.ycombinator.com/user?id=zoogeny) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955492)

I'm not suggesting we re-invent RDF or any other kind of semantic web idea. And the fact that semantic data can be stored in a PDF isn't really the problem being solved by tools such as these. In many cases, PDF is used for things like scanned documents where adding that kind of metadata can't really be done manually - in fact the kinds of tools suggested in the post would be useful for adding that metadata to the PDF after scanning (for example).

Imagine you went to a government office looking for some document from 1930s, like an ancestors marriage or death certificate. You might want to digitize a facsimile of that using a camera or a scanner. You have a lot of options to store that, JPG, PNG, PDF. You have even more options to store the metadata (XML, RDF, TXT, SQLite, etc.). You could even get fancy and zip up an HTML doc alongside a directory of images/resources that stitched them all together. But there isn't really a good standard format to do that.

It is the second part of you post that stands out - the kitchen sink nature of PDFs that make them so terrible. If they were just wrappers for image data, formatted in a way that made printing them easy, I probably wouldn't dislike them.

[groby\_b](https://news.ycombinator.com/user?id=groby_b) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957796)

I mean, you want to store a kitchen sink of data, too. You don't like the semantic web or semantic metadata, fine - what do you propose? A custom metadata format for each use case? That *is* semantic information.

If you don't do that, you get a kitchen sink. If you need to store 1930s death certificats, 10k filings, your doctor's signup forms, the ARR graph for your startup, and a genealogy chart all in the same format, kitchen sink it is.

If it were "just a wrapper for image data", what exactly would that wrapper add? Semantic information, or a kitchen sink to manage additional info.

You're asking to store complex data without preserving complexity - I don't think that'll work.

[zoogeny](https://news.ycombinator.com/user?id=zoogeny) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42965341)

I understand your confusion.

PDF is terrible because it has grown over time from a format that was originally made for one purpose into a format that is used for too many purposes. That organic growth has caused PDFs to be very difficult to use for a wide variety of use cases.

That opinion doesn't imply almost anything else that you have claimed I support (and generally do not).

[layer8](https://news.ycombinator.com/user?id=layer8) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42963020)

Fixed link: [https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandard...](https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/PDF32000_2008.pdf)

[siquick](https://news.ycombinator.com/user?id=siquick) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954594)

Strange that LlamaParse is mentioned in the pricing table but not the results. We’ve used them to process a lot of pages and it’s been excellent each time.

[xena](https://news.ycombinator.com/user?id=xena) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954341)

I really wish that Google made an endpoint that's compatible with the OpenAI API. That'd make trying Gemini in existing flows so much easier.

[myko](https://news.ycombinator.com/user?id=myko) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954457)

I believe this is already the case, at least the Python libraries are compatible, if not recommended for more than just trying things out:

[https://ai.google.dev/gemini-api/docs/openai](https://ai.google.dev/gemini-api/docs/openai)

[msp26](https://news.ycombinator.com/user?id=msp26) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955519)

How well do they work when you want to do things like grounding with search?

[kurtoid](https://news.ycombinator.com/user?id=kurtoid) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954430)

Is that not this? [https://ai.google.dev/api/compatibility](https://ai.google.dev/api/compatibility)

[jonesn11](https://news.ycombinator.com/user?id=jonesn11) [on Feb 13, 2025](https://news.ycombinator.com/item?id=43040790)

OCR makes sense, but it is another asking for a summary. It is not there yet, gave a lot of incorrect details.

[fecal\_henge](https://news.ycombinator.com/user?id=fecal_henge) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953456)

Is there an AI platform where I can paste a snip of a graph and it will generate a n th order polynomial regression for me of the trace?

[CamperBob2](https://news.ycombinator.com/user?id=CamperBob2) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953653)

Either ChatGPT o4 or one of the newer Google models should handle that, since it's a pretty common task. Actually there have been online curve fitters for several years that work pretty well without AI, such as [https://curve.fit/](https://curve.fit/) and [https://www.standardsapplied.com/nonlinear-curve-fitting-cal...](https://www.standardsapplied.com/nonlinear-curve-fitting-calculator.html) .

I'd probably try those first, since otherwise you're depending on the language model to do the right thing automagically.

[potatoman22](https://news.ycombinator.com/user?id=potatoman22) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953982)

I've had decent luck using some of the reasoning models for this. It helps if you task them with identifying where the points on the graph are first.

[kym6464](https://news.ycombinator.com/user?id=kym6464) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961025)

RE: the loss of bounding box information

You can recover word-level bounding boxes and confidence scores by using a traditional OCR engine such as AWS Textract and matching the results to Gemini’s output – see [https://docless.app](https://docless.app) for a demo (disclaimer: I am the founder)

[eichi](https://news.ycombinator.com/user?id=eichi) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42963998)

If is is a vendor work, you should probably hire person who are competitive in software engineering space. And do we actually need significant amount of processing as a solution? If this is the case, common markdowned public pdfs should be open-sourced. We shouldn't repeat other's work.

Despite that, cheaper is better.

[lyjackal](https://news.ycombinator.com/user?id=lyjackal) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958155)

If the end goal is just rag or search over the pdfs, seems like ColPali based embedding search would be a good alternative here. Don’t process the PDFs, instead just search their image embedding directly. From what I understand, you also get a sort of attention as to what part of the image is being activated by the search.

[an\_aparallel](https://news.ycombinator.com/user?id=an_aparallel) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954899)

Has anyone in the AEC industry who's reading this worked out a good way to get Bluebeam MEP, electrical layouts into Revit (LOD 200-300).

Have seen MarkupX as a paid option, but it seems some AI in the loop can greatly speed up exception handling, encode family placement to certain elevations based on building code docs....

[airwaveai](https://news.ycombinator.com/user?id=airwaveai) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957426)

Curious to see how well this works on technical/mechanical documentation (manuals parts list etc). Has any one tried? My company Airwave had to jump through all sorts of hoops to get accurate information for our use case: getting accurate info to the technicians in the field.

[ritvikpandey21](https://news.ycombinator.com/user?id=ritvikpandey21) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956691)

ritvik here from pulse. everyone’s pretty much made the right points here, but wanted to emphasize that due to the llm architecture, they predict “the most probable text string” that corresponds to the embedding, not necessarily the exact text. this non-deterministicness is awful for customers deploying in production and a lot of our customers complained about this to us initially. the best approach is to build a sort-of “agent”-based VLM x traditional layout segmentation/reading order algos, which is what we’ve done and are continuing to do.

we have a technical blog on this exact phenomena coming out in the next couple days, will attach it here when it’s out!

check us out at [https://www.runpulse.com](https://www.runpulse.com)

[bambax](https://news.ycombinator.com/user?id=bambax) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954557)

I'm building a system that does regular OCR and outputs layout-following ASCII; in my admittedly limited tests it works better than most existing offerings.

It will be ready for beta testing this week or the next, and I will be looking for beta testers; if interested please contact me!

[devmor](https://news.ycombinator.com/user?id=devmor) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954446)

I think this is one of the few functional applications of LLMs that is really undeniably useful.

OCR has always been “untrustworthy” (as in you cannot expect it to be 100% correct and know you must account for that) and we have long used ML algorithms for the process.

[nnurmanov](https://news.ycombinator.com/user?id=nnurmanov) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959647)

It is not OCR to blame, when you have garbage in you should not expect anything of high quality, especially with handwriting and tables and different languages. Even human beings fail to understand some documents (see doctor's prescriptions)

[devmor](https://news.ycombinator.com/user?id=devmor) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42963540)

If OCR is a solution designed to recognize documents and it does not recognize all documents, then it is an imperfect solution.

That is not to say there *is* a perfect solution, but it is still the fault of the solution.

[nnurmanov](https://news.ycombinator.com/user?id=nnurmanov) [on Feb 10, 2025](https://news.ycombinator.com/item?id=43001905)

E.g. oftentimes there is l and I (capital I), this may be an issue for OCR. The perfect case is when there is a PDF document and data embedded as XML data, but unfortunately it is not the case.

[sergiotapia](https://news.ycombinator.com/user?id=sergiotapia) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955569)

The article mentions OCR, but you're sending a PDF how is that OCR? Or is this is mistake? What if you send photos of the pages, that would be true OCR - does the performance and price remain the same?

If so this unlocks a massive workflow for us.

[lacoolj](https://news.ycombinator.com/user?id=lacoolj) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42964675)

Anyone know if there are uses of this with PHI? Most doctors still fax reports to each other and this would help a lot to drop the load on staff when receiving and categorizing/assigning to patients

[andrewshadura](https://news.ycombinator.com/user?id=andrewshadura) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961557)

\> Crucially, we’ve seen very few instances where specific numerical values are actually misread.

"Very few" is way too many. This means it cannot be trusted, especially when it comes to financial data.

[jgleoj23](https://news.ycombinator.com/user?id=jgleoj23) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959235)

Gemini is amazing but I get this copyright error for some documents and I have a rate limit of just 10 requests per minute. Same issues with claude except the copyright error is called content warning.

[cedws](https://news.ycombinator.com/user?id=cedws) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953112)

90% accuracy \+/- 10%? What could that be useful for, that’s awfully low.

[lvzw](https://news.ycombinator.com/user?id=lvzw) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953249)

\> accuracy is measured with the Needleman-Wunsch algorithm

\> Crucially, we’ve seen very few instances where specific numerical values are actually misread. This suggests that most of Gemini’s “errors” are superficial formatting choices rather than substantive inaccuracies. We attach examples of these failure cases below \[1\].

\> Beyond table parsing, Gemini consistently delivers near-perfect accuracy across all other facets of PDF-to-markdown conversion.

That seems fairly useful to me, no? Maybe not for mission critical applications, but for a lot of use cases, this seems to be good enough. I'm excited to try these prompts on my own later.

[schainks](https://news.ycombinator.com/user?id=schainks) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953392)

This is "good enough" for Banks to use when doing due diligence. You'd be surprised how much noise is in the system with the current state of the art: algorithms/web scrapers and entire buildings of humans in places like India.

[ai-christianson](https://news.ycombinator.com/user?id=ai-christianson) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953442)

It's certainly pretty useful for discovery/information filtering purposes. I.e. searching for signal in the noise if you have a large dataset.

[jjtheblunt](https://news.ycombinator.com/user?id=jjtheblunt) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953592)

due diligence of this sort?

[https://en.wikipedia.org/wiki/Know\_your\_customer](https://en.wikipedia.org/wiki/Know_your_customer)

[schainks](https://news.ycombinator.com/user?id=schainks) [on Feb 18, 2025](https://news.ycombinator.com/item?id=43085822)

No, I mean services like Bloomberg.

KYC is an API you can pay for now. Works pretty well for the price, IIRC over 10k/month or something.

[raunakchowdhuri](https://news.ycombinator.com/user?id=raunakchowdhuri) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953853)

would encourage you to take a look at some of the real data here! [https://huggingface.co/spaces/reducto/rd\_table\_bench](https://huggingface.co/spaces/reducto/rd_table_bench)

you'll find that most of the errors here are structural issues with the table or inability to parse some special characters. tables can get crazy!

[serjester](https://news.ycombinator.com/user?id=serjester) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953775)

Author here — measuring accuracy in table parsing is surprisingly challenging. Subtle, almost imperceptible differences in how a table is parsed may not affect the reader's understanding but can significantly impact benchmark performance. For all practical purposes, I'd say it's near perfect (also keep in mind the benchmark is on *very* challenging tables).

[summerlight](https://news.ycombinator.com/user?id=summerlight) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953490)

I guess 90% is for "benchmark", which is typically tailored to be challenging to parse.

[mattnewton](https://news.ycombinator.com/user?id=mattnewton) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954057)

having seen some of these tables, I would guess that's probably above a layperson's score . Some are very complicated or just misleadingly structured.

[MattDaEskimo](https://news.ycombinator.com/user?id=MattDaEskimo) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953432)

Switching from manual data entry to approval

[mateuszbuda](https://news.ycombinator.com/user?id=mateuszbuda) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956376)

There’s AWS Bedrock Knowledge Base (Amazon proprietary RAG solution) which can digest PDFs and, as far as I tested it on real world documents, it works pretty well and is cost effective.

[dasl](https://news.ycombinator.com/user?id=dasl) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954892)

How does the Gemini OCR perform against non-English language text?

[jibuai](https://news.ycombinator.com/user?id=jibuai) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954782)

I've been working on something similar the past couple months. A few thoughts:

\- A lot of natural chunk boundaries span multiple pages, so you need some 'sliding window' mechanism for the best accuracy.

\- Passing the entire document hurts throughput too much due to the quadratic complexity of attention. Outputs are also much worse when you use too much context.

\- Bounding boxes can be solved by first generating boxes using tradition OCR / layout recognition, then passing that data to the LLM. The LLM can then link it's outputs to the boxes. Unfortunately getting this reliable required a custom sampler so proprietary models like Gemini are out of the question.

[geckel](https://news.ycombinator.com/user?id=geckel) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957429)

How is it for image recognition/classification? OCR can be a huge chunk of the image classification pipeline. Presumably, it works just as well in this domain?

[hnuser435](https://news.ycombinator.com/user?id=hnuser435) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42958726)

Damn, I thought this was about the Gemini protocol*.*

[https://geminiprotocol.net/](https://geminiprotocol.net/)

[cubefox](https://news.ycombinator.com/user?id=cubefox) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953664)

Why is Gemini Flash so much cheaper than other models here?

[mattnewton](https://news.ycombinator.com/user?id=mattnewton) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954081)

probably a mix of economies of scale (google workspace and search are already massive customers of these models meaning the build out is already there), and some efficiency dividends from hardware r&d (google has developed the model and the TPU hardware purpose built to run it almost in parallel)

[mansourdamanpak](https://news.ycombinator.com/user?id=mansourdamanpak) [on Feb 9, 2025](https://news.ycombinator.com/item?id=42989765)

I've built a simple OCR tool with Gemini 2 flash you can test it here :gemini2flashocr.netlify.app

[jeswin](https://news.ycombinator.com/user?id=jeswin) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957465)

We've previously tried Sonnet in our PDF extraction pipelines. It was very, very accurate, gpt-4o did not come close. Its more expensive, however.

[nottorp](https://news.ycombinator.com/user?id=nottorp) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955411)

Will 2.0.1 also change everything?

How about 2.0.2?

How about Llama 13.4.0.1?

This is tiring. It's always the end of the world when they release a new version of some LLM.

[nnurmanov](https://news.ycombinator.com/user?id=nnurmanov) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959633)

prompt and pray, this is my default mode while working with LLMs

[sensecall](https://news.ycombinator.com/user?id=sensecall) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955068)

This is super interesting.

Would this be suitable for ingesting and parsing wildly variable unstructured data into a structured schema?

[applgo443](https://news.ycombinator.com/user?id=applgo443) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42959756)

Why are traditional OCRs better in terms of hallucination and confidence scores?

Can we use logprobs of LLM as confidence scores?

[dfeng](https://news.ycombinator.com/user?id=dfeng) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961315)

Traditional OCRs are trained for a single task: recognize characters. They do this through visual features (and sometimes there's an implicit (or even explicit) "language" model: see [https://arxiv.org/abs/1805.09441](https://arxiv.org/abs/1805.09441)). As such, the extent of their "hallucination", or errors, is when there's ambiguity in characters, e.g. 0 vs O (that's where the implicit language model comes in). Because they're trained with a singular purpose, you would expect their confidence scores (i.e. logprobs) to be well calibrated. Also, depending on the OCR model, you usually do a text detection (get bounding boxes) followed by a text recognition (read the characters), and so it's fairly local (you're only dealing with a small crop).

On the other hand, these VLMs are very generic models – yes, they're trained on OCR tasks, but also a dozen of other tasks. As such, they're really good OCR models, but they tend to be not as well calibrated. We use VLMs at work (Qwen2-VL to be specific), and we don't find it hallucinates that often, but we're not dealing with long documents. I would assume that as you're dealing with a larger set of documents, you have a much larger context, which increases the chances of the model getting confused and hallucinating.

[grandimam](https://news.ycombinator.com/user?id=grandimam) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960211)

Would you recommend using these large models for parsing sensitive data - probably say bank statements etc?

[jwr](https://news.ycombinator.com/user?id=jwr) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957492)

I wish I could do this locally. I don't feel comfortable uploading all of my private documents to Google.

[aravart](https://news.ycombinator.com/user?id=aravart) [on Feb 7, 2025](https://news.ycombinator.com/item?id=42976644)

Does anyone have some fleshed out source code, prompts and all, to try this on Gemini 2.0?

[nbbaier](https://news.ycombinator.com/user?id=nbbaier) [on Feb 8, 2025](https://news.ycombinator.com/item?id=42980583)

Also really interested in this

[KoolKat23](https://news.ycombinator.com/user?id=KoolKat23) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957424)

Okay I just checked/tried this out with my own use case at work and it's insane.

[ady9999](https://news.ycombinator.com/user?id=ady9999) [on Feb 7, 2025](https://news.ycombinator.com/item?id=42968497)

We have been building smaller and more efficient VLMs for document extraction from way before and we are 10x faster than unstructured,reducto (the ocr vendors) with an accuracy of 90%.

P.S. - You can find us here (unsiloed-ai.com) or you can reach out to me on adnan.abbas@unsiloed-ai.com

[iudqnolq](https://news.ycombinator.com/user?id=iudqnolq) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960999)

In what contexts is 0.84 ± 0.16 actually "nearly perfect"?

[kym6464](https://news.ycombinator.com/user?id=kym6464) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42961102)

I think they meant relative to the best *other* approach, which is Reducto’s given that they are the creators of the benchmark:

Reducto's own model currently outperforms Gemini Flash 2.0 on this benchmark (0.90 vs 0.84). However, as we review the lower-performing examples, most discrepancies turn out to be minor structural variations that would not materially affect an LLM’s understanding of the table.

[ratedgene](https://news.ycombinator.com/user?id=ratedgene) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954130)

Is this something we can run locally? if so what's the license?

[xnx](https://news.ycombinator.com/user?id=xnx) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42955662)

Gemini are Google cloud/service models. Gemma are the Google local models.

[ratedgene](https://news.ycombinator.com/user?id=ratedgene) [on Feb 12, 2025](https://news.ycombinator.com/item?id=43030324)

Ok got it, thanks. Is it a direct mapping?

[otabdeveloper4](https://news.ycombinator.com/user?id=otabdeveloper4) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960010)

Well, probably not literally "everything".

[seunosewa](https://news.ycombinator.com/user?id=seunosewa) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42966003)

He found the one thing that Gemini does better.

[throw7381](https://news.ycombinator.com/user?id=throw7381) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956806)

For data extraction from long documents (100k\+ tokens) how does structured outputs via providing a json schema compare vs asking one question per field (in natural language)?

Also I've been hearing good things regarding document retrieval about Gemini 1.5 Pro, 2.0 Flash and gemini-exp-1206 (the new 2.0 Pro?), which is the best Gemini model for data extraction from 100k tokens?

How do they compare against Claude Sonnet 3.5 or the OpenAI models, has anyone done any real world tests?

[lifeisstillgood](https://news.ycombinator.com/user?id=lifeisstillgood) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42956948)

Imagine there's no PostScript

It's easy if you try

No pdfs below us

Above us only SQL

Imagine all the people livin' for CSV

[mchadda\_chunkr](https://news.ycombinator.com/user?id=mchadda_chunkr) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42960774)

Hi all - CEO of chunkr.ai here.

The write-up and ensuing conversation are really exciting. I think out of everything mentioned here - the clear stand-out point is that document layout analysis (DLA) is the crux of the issue for building practical doc ingestion for RAG.

(Note: DLA is the process of identifying and bounding specific segments of a document - like section headers, tables, formulas, footnotes, captions, etc.)

Strap in - this is going to be a longy.

We see a lot of people and products basically sending complete pages to LVLMs for converting to a machine-readable format, and for chunking. We tried this \+ it’s a possible configuration on chunkr as well. It has never worked for our customers, or during extensive internal testing across documents from a variety of verticals. Here are SOME of the common problems:

\- Most documents are dense. The model will not OCR everything and miss crucial parts.

\- A bunch of hallucinated content thats tough to catch.

\- Occasionally it will just refuse to give you anything. We’ve tried a bunch of different prompting techniques and the models return “\<image>” or “||..|..” for an ENTIRE PAGE of content.

Despite this - it’s obvious that these ginormous neural nets are great for complex conversions like tables and formulas to HTML/Markdown & LateX. They also work great for describing images and converting charts to tables. But that’s the thing - they can only do this if you can pull out these document features individually as cropped images and have the model focus on small snippets of the document rather than the full page.

If you want knobs for speed, quality, and cost, the best approach is to work at a segment level rather than a page level. This is where DLA really shines - the downstream processing options are vast and can be fit to specific needs. You can choose what to process with simple \+ fast OCR (text-only segments like headers, paragraphs, captions), and what to send to a large model like Gemini (complex segments like tables, formulas, and images) - all while getting juicy bounding boxes for mapping citations. Combine this with solid reading order algos - and you get amazing layout-aware chunking that takes \~10ms.

We made RAG apps ourselves and attempted to index all \~600 million pages of open-access research papers for [https://lumina.sh](https://lumina.sh). This is why we built Chunkr - and it needed to be Open Source. You can self-host our solution and process 4 pages per second, scaling up to 11 million pages per month on a single RTX 4090, renting this hardware on Runpod costs just $249/month ($0.34/hour).

A VLM to do DLA sounds awesome. We've played around with this idea but found that VLMs don't come close to models where the architecture is solely geared toward these specific object detection tasks. While it would simplify the pipeline, VLMs are significantly slower and more resource-hungry - they can't match the speed we achieve on consumer hardware with dedicated models. Nevertheless, the numerous advances in the field are very exciting - big if true!

A note on costs:

There are some discrepancies between the API pricing of providers listed in this thread. Assuming 100000 pages \+ feature parity:

Chunkr API - 200 pages for $1, not 100 pages

AWS Textract - 40 pages for $1, not 1000 pages (No VLMs)

Llama Parse - 13 pages for $1, not 300

A note on RD-Bench:

We’ve been using Gemini 1.5 Pro for tables and other complex segments for a while, so the RD-bench is very outdated. We ran it again on a few hundred samples and got a 0.81 (also includes some notes on the bench itself). To the OP: it would be awesome if you could update your blog post!

[https://github.com/lumina-ai-inc/chunkr-table-rdbench/tree/m...](https://github.com/lumina-ai-inc/chunkr-table-rdbench/tree/main)

[killer-Xbox](https://news.ycombinator.com/user?id=killer-Xbox) [on Feb 8, 2025](https://news.ycombinator.com/item?id=42980388)

Hi

[sho\_hn](https://news.ycombinator.com/user?id=sho_hn) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953834)

Remember all the hyperbole a year ago on how Google was failing and over?

[latexr](https://news.ycombinator.com/user?id=latexr) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954275)

Anyone who cries “\<service> is dead” after some new technology is introduced is someone you can safely ignore. For ever. They’re hyperbolic clout chasers who will only ever be right by mistake.

As if, when ChatGPT was introduced, Google would just stay still, cross their arms, and say “well, this is based on our research paper but there’s nothing we can do, going to just roll over and wait for billions of dollars to run out, we’re truly doomed”. So unbelievably stupid.

[throwaway31412](https://news.ycombinator.com/user?id=throwaway31412) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42957664)

fds

[ein0p](https://news.ycombinator.com/user?id=ein0p) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953720)

\> Why Gemini 2.0 Changes Everything

Clickbait. It doesn't change "everything". It makes ingestion for RAG much less expensive (and therefore feasible in a lot more scenarios), at the expense of \~7% reduction in accuracy. Accuracy is already rather poor even before this, however, with the top alternative clocking in at 0.9. Gemini 2.0 is 0.84, although the author seems to suggest that the failure modes are mostly around formatting rather than e.g. mis-recognition or hallucinations.

TL;DR: is this exciting? If you do RAG, yes. Does it "change everything" nope. There's still a very long way to go. Protip for model designers: accuracy is always in greater demand than performance. A slow model that solves the problem is invariably better than a fast one that fucks everything up.

[rvz](https://news.ycombinator.com/user?id=rvz) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954521)

In this use-case, accuracy is non-negotiable with zero room for any hallucination.

Overall it changes nothing.

[ein0p](https://news.ycombinator.com/user?id=ein0p) [on Feb 6, 2025](https://news.ycombinator.com/item?id=42965473)

And people always have a hard time understanding what a certain degree of accuracy actually means. E.g. when you hear that a speech recognition system has 95% accuracy (5% WER), it means that it gets every 19th word wrong. That's abysmally bad by human standards - errors in every other sentence. That does not mean it's useless, but you do need to understand very clearly what you're dealing with, and what those errors might do to the rest of your system.

[nothrowaways](https://news.ycombinator.com/user?id=nothrowaways) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954188)

Cool

[pockmarked19](https://news.ycombinator.com/user?id=pockmarked19) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953835)

Now, I could look at this relatively popular post about Google and revise my opinion of HN as an echo chamber, but I’m afraid it’s just that the downvote loving HNers weren’t able to make the cognitive leap from Gemini to Google.

[raunakchowdhuri](https://news.ycombinator.com/user?id=raunakchowdhuri) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954289)

CTO of Reducto here. Love this writeup!

We’ve generally found that Gemini 2.0 is a great model and have tested this (and nearly every VLM) very extensively.

A big part of our research focus is incorporating the best of what new VLMs offer without losing the benefits and reliability of traditional CV models. A simple example of this is we’ve found bounding box based attribution to be a non-negotiable for many of our current customers. Citing the specific region in a document where an answer came from becomes (in our opinion) even MORE important when using large vision models in the loop, as there is a continued risk of hallucination.

Whether that matters in your product is ultimately use case dependent, but the more important challenge for us has been reliability in outputs. RD-TableBench currently uses a single table image on a page, but when testing with real world dense pages we find that VLMs deviate more. Sometimes that involves minor edits (summarizing a sentence but preserving meaning), but sometimes it’s a more serious case such as hallucinating large sets of content.

The more extreme case is that internally we fine tuned a version of Gemini 1.5 along with base Gemini 2.0, specifically for checkbox extraction. We found that even with a broad distribution of checkbox data we couldn’t prevent frequent checkbox hallucination on both the flash (\+17% error rate) and pro model (\+8% error rate). Our customers in industries like healthcare expect us to get it right, out of the box, deterministically, and our team’s directive is to get as close as we can to that ideal state.

We think that the ideal state involves a combination of the two. The flexibility that VLMs provide, for example with cases like handwriting, is what I think will make it possible to go from 80 or 90 percent accuracy to some number very close 99%. I should note that the Reducto performance for table extraction is with our pre-VLM table parsing pipeline, and we’ll have more to share in terms of updates there soon. For now, our focus is entirely on the performance frontier (though we do scale costs down with volume). In the longer term as inference becomes more efficient we want to move the needle on cost as well.

Overall though, I’m very excited about the progress here.

--- One small comment on your footnote, the evaluation script with Needlemen-Wunsch algorithm doesn’t actually consider the headers outputted by the models and looks only at the table structure itself.

[noja](https://news.ycombinator.com/user?id=noja) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42954464)

\> deterministically

How are you planning to do this?

[resource\_waste](https://news.ycombinator.com/user?id=resource_waste) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953836)

Google's models have historically been total disappointments compared to chatGPT4. Worse quality, wont answer medical questions either.

I suppose I'll try it again, for the 4th or 5th time.

This time I'm not excited. I'm expecting it to be a letdown.

[coderstartup](https://news.ycombinator.com/user?id=coderstartup) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953634)

Following this post

[exabrial](https://news.ycombinator.com/user?id=exabrial) [on Feb 5, 2025](https://news.ycombinator.com/item?id=42953539)

You know what'd be fucking nice? The ability to turn Gemini off.

[pmarreck](https://news.ycombinator.com/user?id=pmarreck) [on Feb 17, 2025](https://news.ycombinator.com/item?id=43081117)

hAIters gonna hAIte
