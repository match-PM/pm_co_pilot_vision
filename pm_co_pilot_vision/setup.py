from setuptools import find_packages, setup

package_name = 'pm_co_pilot_vision'
modules = "pm_co_pilot_vision/co_pilot_modules"
tools = "pm_co_pilot_vision/co_pilot_modules/tools"
utils = "pm_co_pilot_vision/utils"
files = 'files'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, modules, tools, utils, files],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['files/vision_functions.json']),
        ('share/' + package_name + '/launch', ['launch/pm_co_pilot_vision.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='match-mover',
    maintainer_email='wiemann@match.uni-hannover.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pm_co_pilot_vision = pm_co_pilot_vision.pm_co_pilot_vision:main'
        ],
    },
)
