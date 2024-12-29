from setuptools import setup, find_namespace_packages

setup(
    name='eliana',
    version='0.1.6',
    author='Juan Pablo Gil',
    author_email='juanpablo.gil@eso.org',
    description='ELIANA (Event Log and Incident Analysis), a log analysis library developed at the Paranal Observatory',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/paranal-sw/eliana',
    packages=find_namespace_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',  # Updated to BSD License
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=[
        'pandas',
        'matplotlib',
        'numpy',
        'scikit-learn',
        'pyarrow',
        'fastparquet',
    ],
    include_package_data=True,
    zip_safe=False,  # Recommended for namespace packages
)
