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
    install_requires=['setuptools'],
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
        ],
    },
)
