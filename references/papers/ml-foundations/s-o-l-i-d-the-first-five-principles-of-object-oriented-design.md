Introducing DigitalOcean Managed Agents Runtime Services (Private Preview)
Blog Docs Careers Get Support Contact Sales
Products Solutions Developers Partners Pricing
Log in
Tutorials Questions Product Docs Search Comm
SOLID Design Principles Explained:
Building Better Software
Architecture
Conceptual Articles / Development / SOLID Design Principles Explained: Building
Better Software Architecture
Conceptual Article
Updated on June 11, 2025
Development
Language
By Samuel Oloruntoba, and
Anish Singh Walia Manikandan
Kurup
English
Table of contents
SingleResponsibility
Principle
OpenClosed Principle
Liskov Substitution
Principle
Interface Segregation
Principle
Dependency Inversion
Principle
Frequently Asked
Questions FAQs
Conclusion
Popular topics
Introduction
SOLID is an acronym for the first five object-oriented
design (OOD) principles by Robert C. Martin (also
known as Uncle Bob).
Note: While these principles can apply to various
programming languages, the sample code
contained in this article will use PHP.
These principles establish practices for developing
software with considerations for maintaining and
extending it as the project grows. Adopting these
practices can also help avoid code smells, refactor
code, and develop Agile or Adaptive software.
SOLID stands for:
S - Single-responsibility Principle
O - Open-closed Principle
L - Liskov Substitution Principle
I - Interface Segregation Principle
D - Dependency Inversion Principle
In this article, you will be introduced to each principle
individually to understand how SOLID can help make
you a better developer.
Single-Responsibility Principle
Single-responsibility Principle (SRP) states:
A class should have one and only one
reason to change, meaning that a class
should have only one job.
For example, consider an application that takes a
collection of shapes—circles and squares—and
calculates the sum of the area of all the shapes in the
collection.
First, create the shape classes and have the
constructors set up the required parameters.
For squares, you will need to know the side:
length of a
Copy
class Square
{
public $length;
public function __construct($length)
{
$this->length = $length;
}
}
For circles, you will need to know the radius :
Copy
class Circle
{
public $radius;
public function __construct($radius)
{
$this->radius = $radius;
}
}
Next, create the AreaCalculator class and write the
logic to sum up the areas of all provided shapes. The
area of a square is calculated by length squared, and
the area of a circle is calculated by pi times radius
squared.
Copy
class AreaCalculator
{
protected $shapes;
public function __construct($shapes = [])
{
$this->shapes = $shapes;
$this->shapes = $shapes;
}
{
public function sum()
$area = [];
foreach ($this->shapes as $shape) {
if (is_a($shape, 'Square')) {
$area[] = pow($shape->length, 2);
} elseif (is_a($shape, 'Circle')) {
$area[] = pi() * pow($shape->radius, 2
}
}
return array_sum($area);
}
{
public function output()
return implode(''
''
, [
,
'Sum of the areas of provided shapes: $this->sum(),
',
''
,
]);
}
}
To use the AreaCalculator class, you will need to
instantiate it, pass in an array of shapes, and display
the output at the bottom of the page.
Here is an example with a collection of three shapes:
a circle with a radius of 2
a square with a length of 5
a second square with a length of 6
Copy
$shapes = [
new Circle(2),
new Square(5),
new Square(6),
];
$areas = new AreaCalculator($shapes);
echo $areas->output();
The AreaCalculator class, as designed above, has
several issues regarding the Single Responsibility
Principle:
It directly calculates the area for each specific
shape within its sum() method. This means if
you introduce a new shape (e.g., Triangle), you
have to modify AreaCalculator . This violates
SRP (the AreaCalculator has a reason to
change for every new shape type).
It also handles data output/presentation (e.g.,
HTML, JSON, plain text) within its output()
method. This is a separate concern from
calculation.
Let’s apply SRP in two stages to address these issues.
Stage 1: Giving Shapes Their Own
Responsibility
In a truly object-oriented design and aligning with SRP,
a shape’s single responsibility should include knowing
its own properties and how to perform operations
intrinsically related to those properties, such as
calculating its own area. The AreaCalculator should
not need to know the specific formula for every shape.
To address this, we move the area() calculation logic
into each individual shape class:
Here is the area method defined in Square :
Copy
class Square{
public $length;
public function __construct($length)
{
$this->length = $length;
}
public function area()
{
return pow($this->length, 2);
}
}
And here is the area method defined in Circle :
Copy
class Circle{
public $radius;
public function __construct($radius)
{
$this->radius = $radius;
}
public function area()
{
return pi() * pow($this->radius, 2);
}
}
Now, the sum method for AreaCalculator can be
rewritten to simply ask each shape for its area, without
knowing the specific calculation formula:
Copy
class AreaCalculator{
protected $shapes;
public function __construct($shapes = [])
{
$this->shapes = $shapes;
}
{
public function sum()
$area = [];
foreach ($this->shapes as $shape) {
// Each shape is now responsible for its o
$area[] = $shape->area();
}
return array_sum($area);
}
{
public function output()
// This method still handles output, which we'
return implode(''
, [
''
,
'Sum of the areas of provided shapes: ',
$this->sum(),
''
,
]);
}
}
Now, adding a new shape type (e.g., Triangle ) will
not require modifying the AreaCalculator ’s sum()
method. We’ve applied SRP by giving shapes the
single responsibility of knowing their own area.
Stage 2: Separating Calculation from
Output
Even after Stage 1, the AreaCalculator class still has
two primary responsibilities:
Calculating the sum of areas: This is its core
mathematical function.
Handling data output/presentation: This involves
formatting the sum for display (e.g., HTML,
JSON, plain text).
Consider a scenario where the output should be
converted to another format like JSON. The
AreaCalculator class would currently handle all of
this output logic directly within its output() method.
If we needed a JSON output, we’d add more logic to
output() , or create outputJson() , etc. This means
the AreaCalculator would change not only if the
calculation logic changed, but also if the output format
changed.
The AreaCalculator class should primarily be
concerned with the sum of the areas of provided
shapes and should not care whether the user wants
JSON or HTML.
To address this, you can apply SRP by separating the
concerns. You can create a separate
SumCalculatorOutputter class (or classes) and use
that new class to handle the logic you need to output
the data to the user:
Copy
class SumCalculatorOutputter
{
protected $calculator;
public function __construct(AreaCalculator $calcul
{
$this->calculator = $calculator;
$this->calculator = $calculator;
}
{
public function JSON()
$data = [
'sum' => $this->calculator->sum(),
];
return json_encode($data);
}
{
public function HTML()
return implode(''
''
, [
,
'Sum of the areas of provided shapes: $this->calculator->sum(),
',
''
,
]);
}
}
The SumCalculatorOutputter class would work like
this:
Copy
$shapes = [
new Circle(2),
new Square(5),
new Square(6),
];
$areas = new AreaCalculator($shapes);
$output = new SumCalculatorOutputter($areas);
echo $output->JSON();
echo $output->HTML();
Now, the AreaCalculator class has a single
responsibility: calculating the sum of areas and
SumCalculatorOutputter has a single responsibility
of formatting and outputting calculation results. This
separation satisfies the Single Responsibility Principle
for these two classes.
Open-Closed Principle
Open-closed Principle (OCP) states:
Objects or entities should be open for
extension but closed for modification.
This means that a class should be extendable without
modifying the class itself.
Let’s revisit the AreaCalculator class and focus on
the sum method:
Copy
class AreaCalculator
{
protected $shapes;
public function __construct($shapes = [])
{
$this->shapes = $shapes;
}
{
public function sum()
foreach ($this->shapes as $shape) {
if (is_a($shape, 'Square')) {
$area[] = pow($shape->length, 2);
} elseif (is_a($shape, 'Circle')) {
$area[] = pi() * pow($shape->radius, 2
}
}
}
return array_sum($area);
}
}
Consider a scenario where the user would like the
sum of additional shapes like triangles, pentagons,
hexagons, etc. You would have to constantly edit this
file and add more if / else blocks. That would
violate the open-closed principle.
A way you can make this sum method better is to
remove the logic to calculate the area of each shape
out of the AreaCalculator class method and attach it
to each shape’s class.
Here is the area method defined in Square :
Copy
class Square
{
public $length;
public function __construct($length)
{
$this->length = $length;
}
public function area()
{
}
return pow($this->length, 2);
}
And here is the area method defined in Circle :
Copy
class Circle
{
public $radius;
public function __construct($radius)
{
$this->radius = $radius;
}
public function area()
{
return pi() * pow($this->radius, 2);
}
}
The sum method for AreaCalculator can then be
rewritten as:
Copy
class AreaCalculator
{
// ...
public function sum()
{
foreach ($this->shapes as $shape) {
$area[] = $shape->area();
}
return array_sum($area);
}
}
Now, you can create another shape class and pass it in
when calculating the sum without breaking the code.
However, another problem arises. How do you know
that the object passed into the AreaCalculator is
actually a shape or if the shape has a method named
area ?
Coding to an interface is an integral part of SOLID.
Create a ShapeInterface that supports area :
Copy
interface ShapeInterface
{
}
public function area();
Modify your shape classes to implement the
ShapeInterface.
Here is the update to Square :
Copy
class Square implements ShapeInterface
{
}
// ...
And here is the update to Circle :
Copy
class Circle implements ShapeInterface
{
// ...
}
In the sum method for AreaCalculator , you can
check if the shapes provided are actually instances of
the ShapeInterface ; otherwise, throw an exception:
Copy
{
class AreaCalculator
// ...
public function sum()
{
foreach ($this->shapes as $shape) {
if (is_a($shape, 'ShapeInterface')) {
$area[] = $shape->area();
continue;
}
throw new AreaCalculatorInvalidShapeExcept
}
return array_sum($area);
}
}
That satisfies the open-closed principle.
Liskov Substitution Principle
Liskov Substitution Principle states:
Let q(x) be a property provable about
objects of x of type T. Then q(y) should be
provable for objects y of type S where S is
a subtype of T.
This means that every subclass or derived class
should be substitutable for their base or parent class.
Building off the example AreaCalculator class,
consider a new VolumeCalculator class that extends
the AreaCalculator class:
Copy
class VolumeCalculator extends AreaCalculator
{
public function __construct($shapes = [])
{
parent::__construct($shapes);
}
{
public function sum()
// logic to calculate the volumes and then ret
return [$summedData];
}
}
Recall that the SumCalculatorOutputter class
resembles this:
Copy
class SumCalculatorOutputter {
protected $calculator;
public function __construct(AreaCalculator $calcul
$this->calculator = $calculator;
}
public function JSON() {
$data = array(
'sum' => $this->calculator->sum(),
'sum' => $this->calculator->sum(),
);
return json_encode($data);
}
public function HTML() {
return implode(''
''
, array(
,
'Sum of the areas of provided shapes:
$this->calculator->sum(),
''
));
}
}
If you tried to run an example like this:
Copy
$areas = new AreaCalculator($shapes);
$volumes = new VolumeCalculator($solidShapes);
$output = new SumCalculatorOutputter($areas);
$output2 = new SumCalculatorOutputter($volumes);
When you call the HTML method on the $output2
object, you will get an E_NOTICE error, informing you
of an array-to-string conversion.
To fix this, instead of returning an array from the
VolumeCalculator class sum method, return
$summedData :
Copy
class VolumeCalculator extends AreaCalculator
{
public function __construct($shapes = [])
public function __construct($shapes = [])
parent::__construct($shapes);
{
}
{
public function sum()
// logic to calculate the volumes and then ret
return $summedData;
}
}
The integer.
$summedData can be a float, a double or an
That satisfies the Liskov substitution principle.
Interface Segregation Principle
The interface segregation principle states:
A client should never be forced to
implement an interface that it doesn’t use,
or clients shouldn’t be forced to depend on
methods they do not use.
This principle emphasizes that large, general-purpose
interfaces should be broken down into smaller, more
specific ones. This way, client classes only need to
know about the methods that are relevant to them.
Continuing from the previous ShapeInterface
example, let’s consider a scenario where you need to
support new three-dimensional shapes of Cuboid and
Spheroid , and these shapes will need to also
calculate both area and volume.
Let’s consider what would happen if you were to
modify the ShapeInterface to add another contract:
Copy
interface ShapeInterface
{
public function area();
public function volume();
}
Now, any shape you create must implement the
volume method, but you know that squares are flat
shapes and that they do not have volumes, so this
interface would force the Square class to implement a
method that it doesn’t need.
This would violate the interface segregation principle.
Instead of having one large, monolithic interface, we
should create separate, more granular interfaces that
define specific capabilities.
We keep ShapeInterface for two-dimensional shapes
that only have an area:
Copy
interface ShapeInterface
{
}
public function area();
And we create a new interface specifically for three-
dimensional shapes that can calculate volume:
Copy
interface ThreeDimensionalShapeInterface
{
public function volume();
}
Now, concrete shape classes can implement only the
interfaces that are relevant to their capabilities.
For 2D shapes:
Copy
class Square implements ShapeInterface { // Only imple
public $length;
public function __construct($length) {
$this->length = $length;
}
public function area() {
return pow($this->length, 2);
}
}
For 3D Shapes (e.g., Cuboid) which have both surface
area and volume, the class implements both relevant
interfaces.
Copy
class Cuboid implements ShapeInterface, ThreeDimension
{
public function area()
{
// calculate the surface area of the cuboid
// calculate the surface area of the cuboid
}
{
}
public function volume()
// calculate the volume of the cuboid
}
The Square class (and any other 2D shape) is no
longer forced to implement a volume() method it
doesn’t need. It only implements ShapeInterface and
its area() method.
This approach ensures that clients are not forced to
depend on interfaces (or methods within interfaces)
that they do not use, leading to cleaner, more cohesive
code.
Dependency Inversion Principle
Dependency inversion principle states:
Entities must depend on abstractions, not
on concretions. It states that the high-level
module must not depend on the low-level
module, but they should depend on
abstractions.
This principle allows for decoupling.
Here is an example of a PasswordReminder that
connects to a MySQL database:
Copy
class MySQLConnection
{
public function connect()
{
// handle the database connection
return 'Database connection';
}
}
{
class PasswordReminder
private $dbConnection;
public function __construct(MySQLConnection $dbCon
{
$this->dbConnection = $dbConnection;
}
}
First, the MySQLConnection is the low-level module
while the PasswordReminder is high level, but
according to the definition of D in SOLID, which states
to Depend on abstraction, not on concretions. This
snippet above violates this principle as the
PasswordReminder class is being forced to depend on
the MySQLConnection class.
Later, if you were to change the database engine (e.g.,
from MySQL to PostgreSQL or an API service), you
would also have to edit the PasswordReminder class,
which would violate the open-close principle , as
the class would need modification for extension.
The PasswordReminder class should not care what
database your application uses. To address these
issues, you can code to an interface since high-level
and low-level modules should depend on abstraction.
First, define an interface for database connections.
This interface ( DBConnectionInterface ) serves as the
abstraction:
Copy
interface DBConnectionInterface
{
public function connect();
}
The interface has a connect method, and the
MySQLConnection class implements this interface.
Also, instead of directly type-hinting the
MySQLConnection class in the constructor of the
PasswordReminder , you type-hint the
DBConnectionInterface . No matter the type of
database your application uses, the
PasswordReminder class can connect to the database
without any problems, and the open-close principle is
not violated.
This interface simply declares what a database
connection object should be able to do ( connect() ),
without specifying how it does it.
Next, your concrete MySQLConnection class will
implement this interface, providing the specific details
of how to connect to a MySQL database:
Copy
class MySQLConnection implements DBConnectionInterface
{
public function connect()
{
// handle the database connection
return 'Database connection established';
}
}
}
Now, in the constructor of the PasswordReminder
class, instead of directly type-hinting the concrete
MySQLConnection class, you type-hint the
DBConnectionInterface.
Copy
class PasswordReminder{
private $dbConnection;
public function __construct(DBConnectionInterface
{
$this->dbConnection = $dbConnection;
}
public function remind() {
$connectionStatus = $this->dbConnection->conne
return "Password reminder process initiated. C
}
}
When you use PasswordReminder , you provide it with
an instance of a class that implements
DBConnectionInterface . The PasswordReminder
doesn’t need to know the specific type of connection
(MySQL, PostgreSQL, etc.); it just knows it has an
object that guarantees it can call a connect()
method.
Here’s how you would use it in your application:
Copy
// Create a concrete MySQL connection object
$mysqlConnector = new MySQLConnection();
// Inject the concrete MySQL connection object into th
// The PasswordReminder only sees it as a DBConnection
$passwordReminder = new PasswordReminder($mysqlConnect
echo $passwordReminder->remind(); // Output: Password
If you later decide to switch to a PostgreSQL database,
you would simply create a PostgreSQLConnection
class that also implements DBConnectionInterface
and, in your application’s setup, simply change which
concrete class you instantiate and inject:
Copy
$pgConnector = new PostgreSQLConnection();
$passwordReminder = new PasswordReminder($pgConnector)
echo $passwordReminder->remind();
This demonstrates how easily you can swap
implementations thanks to dependency inversion.
Notice that the PasswordReminder class itself did not
need to be changed or modified when the underlying
database technology changed. Both the high-level
PasswordReminder module and the low-level
MySQLConnection (or PostgreSQLConnection ) module
now depend on the DBConnectionInterface
abstraction, not on each other’s concretions. This
makes the system more flexible, easier to test (you can
inject mock database connections), and adheres to the
Open-Closed Principle.
Frequently Asked Questions
(FAQs)
1. What are the 5 SOLID principles in
software engineering?
SOLID is an acronym representing five fundamental
object-oriented design principles formulated by Robert
C. Martin, also known as Uncle Bob. These principles
are:
Single-responsibility Principle (SRP): A class
should have one and only one reason to change.
Open-closed Principle (OCP): Software entities
(classes, modules, functions, etc.) should be
open for extension, but closed for modification.
Liskov Substitution Principle (LSP): Objects in a
program should be replaceable with instances of
their subtypes without altering the correctness of
that program.
Interface Segregation Principle (ISP): Clients
should not be forced to depend on interfaces
they do not use.
Dependency Inversion Principle (DIP): High-
level modules should not depend on low-level
modules; both should depend on abstractions.
Also, abstractions should not depend on details;
details should depend on abstractions.
Together, these principles establish a set of best
practices for developing software that is robust,
adaptable, and easy to maintain and extend as
projects evolve.
2. Why is SOLID important in object-
oriented programming?
SOLID principles are critically important in object-
oriented programming because they directly address
common challenges in software development, such as
rigidity, fragility, immobility, and viscosity. By adhering
to SOLID, developers can build systems that are:
More Maintainable: Easier to understand and fix
bugs.
More Flexible: Can adapt to changing
requirements without significant re-architecting.
More Extensible: New features can be added
with minimal impact on existing code.
Easier to Test: Components can be isolated and
tested independently, reducing the risk of side
effects.
Less Prone to Code Smells: They guide towards
cleaner, more organized codebases, preventing
the accumulation of bad design choices.
Ultimately, SOLID helps in creating high-quality
software that is easier to evolve and has a longer
lifespan.
3. How do I apply the Single Responsibility
Principle?
Applying the Single Responsibility Principle (SRP)
involves understanding the “responsibilities” of a class
or module. If you can identify more than one distinct
“reason to change” for a given class, it likely violates
SRP. For instance, a class that both processes data
and saves it to a database has two reasons to change:
one if the data processing logic changes, and another
if the database interaction method changes.
To apply SRP, you should:
Identify distinct responsibilities: Ask yourself,
“What would cause this class to change?” If there
are multiple, independent answers, those are
separate responsibilities.
Extract responsibilities into separate
classes/modules: Create new, focused classes
or modules for each identified responsibility.
Ensure each new entity has only one reason to
change: The goal is that a modification in
requirements for one responsibility should only
affect one class, not multiple.
This separation leads to smaller, more focused, and
highly cohesive components that are easier to
understand, test, and maintain.
4. What is the difference between Open-
Closed Principle (OCP) and the
Dependency Inversion Principle (DIP)?
While both the Open-Closed Principle (OCP) and the
Dependency Inversion Principle (DIP) aim to reduce
coupling and increase flexibility, they address different
aspects of design:
Open-Closed Principle (OCP) focuses on
behavioral extension. It dictates that existing
code should not be modified when new
functionality is added. Instead, you should extend
its behavior, typically by creating new classes
that implement an interface or extend an abstract
class, or by using composition. The emphasis is
on adding features by adding new code, not by
changing old code.
Dependency Inversion Principle (DIP) focuses on
dependency direction and abstraction. It states
that high-level modules (which contain important
business logic) should not depend on low-level
modules (which handle details like database
interaction or file I/O). Instead, both should
depend on abstractions (interfaces or abstract
classes). This “inverts” the typical top-down
dependency flow, promoting a system where
concrete details depend on abstract contracts.
In essence, DIP is often a key enabler for OCP. By
depending on abstractions (DIP), you can easily
substitute different concrete implementations, allowing
you to extend functionality (OCP) without altering the
core logic that depends on those abstractions. OCP is
a design goal, and DIP is a powerful pattern to achieve
that goal.
5. Are SOLID principles only for OOP?
While the SOLID principles were originally articulated
in the context of Object-Oriented Programming (OOP)
and use terms like “class” and “interface,” their
underlying philosophies and benefits extend far
beyond strict OOP. The core ideas of managing
dependencies, isolating changes, promoting
modularity, and enabling extensibility are universal to
good software design.
The Single Responsibility Principle can be
applied to functions, modules, microservices, or
even entire teams.
The idea of open for extension, closed for
modification (OCP) is desirable in any
architectural style.
Liskov Substitution (LSP) applies whenever you
have polymorphic relationships, regardless of the
language’s specific features.
Interface Segregation (ISP) promotes breaking
down large contracts into smaller, more client-
specific ones, which is valuable even in
functional programming or service design.
Dependency Inversion (DIP) encourages
decoupling through abstractions, a crucial
concept in many architectural patterns like
hexagonal architecture or clean architecture,
which aren’t exclusively OOP. Therefore, while
their origins are in OOP, the SOLID principles
offer timeless wisdom for designing robust and
maintainable software systems across various
paradigms and scales.
Conclusion
By now, you’ve gained a solid understanding of the
SOLID principles. These aren’t just abstract concepts;
they’re powerful tools for improving the design of your
object-oriented systems. Implementing them
consistently leads to code that’s more maintainable,
easier to extend, simpler to test, and less prone to
breaking when requirements shift. Master SOLID, and
you’ll elevate your development skills, creating
software that stands the test of time.
Continue your learning by reading about other
practices for Agile vs. Waterfall and Most Common
Design Patterns in Java.
Thanks for learning with the DigitalOcean
Community. Check out our offerings for
compute, storage, networking, and managed
databases.
Learn more about our products
About the author(s)
Samuel Oloruntoba Author
Anish Singh Walia Editor
Sr Technical Content Strategist
and Team Lead
See author
profile
Anish is a Sr Technical Content Strategist and Team
Lead at DigitalOcean with 7+ years of experience as
an DevOps SRE at Nutanix and Cloud consultant at
AMEX, and technical writing at DOCN, and shipping
deep infra and AI inference tutorials that help AI-
Native Enterprises and teams deploy production-ready
applications on DigitalOcean.
Manikandan
Kurup
Senior Technical Content
Engineer I
Editor
See author
profile
With over 6 years of experience in tech publishing,
Mani has edited and published more than 75 books
covering a wide range of data science topics. Known
for his strong attention to detail and technical
knowledge, Mani specializes in creating clear, concise,
and easy-to-understand content tailored for
developers.
Category:
Conceptual
Article
Tags: Development
Still looking for an answer?
Ask a question Search for more help
Was this helpful? Yes No
Comments(10) Follow-up questions(0)
Leave a comment...
This textbox defaults to using Markdown to format your
answer.
You can type !ref in this text area to quickly search our full set
of tutorials, documentation & marketplace offerings and insert
the link!
Sign in/up to comment
webitbest
Show less
November 23, 2020
November 23, 2020
Would you kindly clear dependency inversion
for me. As I caught it, in the first example real
MySQLConnection instance injected into
PasswordReminder. But if we inject interface,
which is actually nothing, will things work? Or
we will have to implement it later? What’s then
use of all this stuff?
Reply (2) replies
adamrabczuk
February 24, 2021
This comment has been deleted
adamrabczuk
February 24, 2021
This comment has been deleted
adamrabczuk
February 24, 2021
This comment has been deleted
adamrabczuk
February 24, 2021
This comment has been deleted
adamrabczuk
February 24, 2021
This comment has been deleted
adamrabczuk
February 24, 2021
This comment has been deleted
adamrabczuk
Show less
February 24, 2021
IMO the examples are not thought through.
For example: does it make sense to have an
object that is of
ThreeDimensionalShapeInterface type and is
not of type ShapeInterface? And the
proposition here is to make developer
remember to always implement the third
interface ManageShapeInterface.
Reply
adamrabczuk
February 24, 2021
This comment has been deleted
adamrabczuk
February 24, 2021
This comment has been deleted
Load more comments
This work is licensed under a Creative Commons
Attribution-NonCommercial- ShareAlike 4.0
International License.
Become a
contributor
for
DigitalOcean
Documentation
Full
Resources
for
startups
community
and AI-
community
Get paid to write
technical
tutorials and
select a tech-
focused charity
to receive a
matching
donation.
documentation
for every
DigitalOcean
product.
and AI-
native
businesses
The Wave has
everything you
need to know
about building a
business, from
raising funding to
marketing your
product.
Sign Up
Learn more
Learn more
Stay up to date, sign up for our
newsletter
Email*
Email address
Submit
The
developer
cloud
Scale up as you grow —
whether you're running one
virtual machine or ten
thousand.
Start building
today
From GPU-powered inference
and Kubernetes to managed
databases and storage, get
everything you need to build,
scale, and deploy intelligent
applications.
View all products
Sign up
Company Products Resources
About
Leadership
Blog
Careers
Customers
Partners
Referral Program
Affiliate Program
Press
Legal
Privacy Policy
Security
Investor Relations
Knowledge Bases
GPU Droplets
Bare Metal GPUs
Inference Engine
Data & Learning
Evaluations
Model Library
Droplets
Kubernetes
Functions
App Platform
Load Balancers
Managed Databases
Spaces
Block Storage
Network File Storage
API
Uptime
Cloud Security Posture
Management (CSPM)
Identity and Access
Management (IAM)
Cloudways
View all Products
Solutions Contact
Support
Sales
Report Abuse
System Status
Share your ideas
Community Tutorials
Community Q&A
CSS-Tricks
Write for DOnations
Currents Research
DigitalOcean Startups
Wavemakers Program
Compass Council
Open Source
Newsletter Signup
Marketplace
Pricing
Pricing Calculator
Documentation
Release Notes
Code of Conduct
Shop Swag
AI Training GPU
GPU Inference
VPS Hosting
Website Hosting
VPN
Docker Hosting
Node.js Hosting
Web Mobile Apps
WordPress Hosting
Virtual Machines
View all Solutions
© 2026 DigitalOcean,
LLC. Sitemap.
This site uses cookies and related technologies, as described in
our privacy policy, for purposes that may include site operation,
analytics, enhanced user experience, or advertising. You may choose
to consent to our use of these technologies, or manage your own
preferences. Please visit our cookie policy for more information.
AGREE & PROCEED
DECLINE ALL
MANAGE CHOICES
Cookie Preferences
