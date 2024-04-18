import setuptools

__version__ = "0.0.0"

REPO_NAME = "Named-Entity-Recognition-News"
AUTHOR_USER_NAME = "Sajith17"
SRC_REPO = "NamedEntityRecognition"
AUTHOR_EMAIL = "sajithseelan17@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A python package for NER app",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)