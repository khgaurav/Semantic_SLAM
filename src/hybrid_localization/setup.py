from setuptools import find_packages, setup

package_name = 'hybrid_localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'faiss-cpu==1.7.4',
        'numpy',
        'Pillow>=9.1.0',
        'sentencepiece',
        'torch',
        'transformers',
    ],
    zip_safe=True,
    maintainer='gauravkh',
    maintainer_email='kothamachuharish.g@northeastern.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mapping_node = hybrid_localization.mapping_node:main',
            'localization_node = hybrid_localization.localization_node:main',
            (
                'compressed_image_republisher = '
                'hybrid_localization.compressed_image_republisher:main'
            ),
        ],
    },
)
