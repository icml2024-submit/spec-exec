from setuptools import setup, find_packages

setup(
    name="specdec",
    version="0.1",
    package_dir={"": "specdec"},
    packages=find_packages("specdec"),
    description="Companion code for SpecExec paper and beyond",
    keywords="speculative decoding",
    install_requires=[
        'torch >= 2.1',
        'optimum >= 1.16.1',
        'auto-gptq >= 0.6.0',
        'transformers >= 4.37.2',
        'anytree >= 2.12.0',  # for tree charting in debug mode
        'matplotlib >= 3.7',
        'seaborn >= 0.13.1',
        'numpy >= 1.24',
        'pandas >= 2.0.3',
        'notebook >= 7.0.6',
    ],
    python_requires='>=3.8',

    # If you have scripts that should be directly callable from the command line, you can specify them here.
    scripts=['run_exp.py'],
    package_data={
        # Include any non-code files in your package
        'oasst_prompts': ['data/oasst_prompts.json'],
        'wikitext_prompts': ['data/wikitext_prompts.json'],
    },
    include_package_data=True,
)
