{
    "input_data_location": "E:\\Public_interest_project\\Data\\input\\",
    "output_data_location": "A:\\Phoenix_project_Output\\",
    "subdirectories": {
        "processed_data": "A:\\Phoenix_project_Output\\processed_data\\",
        "visualizations": "A:\\Phoenix_project_Output\\visualizations\\",
        "logs": "A:\\Phoenix_project_Output\\logs\\",
        "reports": "A:\\Phoenix_project_Output\\reports\\"
    },
    "directories": {
        "scripts_module": "E:\\Public_interest_Project\\Scripts_Module\\",
        "scripts_test": "E:\\Public_interest_Project\\Scripts_Test\\",
        "config_files": "E:\\Public_interest_Project\\config_files\\",
        "data": "E:\\Public_interest_Project\\Data\\",
        "models_source": "E:\\Public_interest_Project\\Models_Source\\",
        "venv_openvino": "E:\\Public_interest_Project\\venv_openvino\\",
        "venv_project_002": "E\\Public_interest_Project\\Venv_Project_002\\",
        "venv_project_phoenix": "E\\Public_interest_Project\\Venv_Project_Phoenix\\"
    },
{

https://icd.who.int/docs/icd-api/ReleaseNotes-Version2.4/
Installing ICD-API as a Windows service located at "C:\Program Files\ICD-API\ICD-API.exe"
Windows services are programs that operate in the background. It is possible to install the ICD-API as a Windows service.

Prerequisites
The binary files provided for Windows are x64 binaries and requires an x64 Windows installation
By installing this package you agree with our license agreement
Steps for Installing the ICD-API as a Windows service
Download the ICD-API as Windows service version 2.4.0 installation package
Start the installer by double clicking on the downloaded file icdapi-setup-2.4.0.msi
Follow the instructions on the screen.
If you select the saveAnalytics option, the software that is installed can send data to WHO on the searches made in order to improve search capability of ICD-11 tools. The data that is sent does not contain any ids or IP addresses.
Checking the Installation
By default the API is installed at port 8382 so you should be able to access the API from http://localhost:8382/icd/... from the same computer.
IMPORTANT! Please note that in earlier releases we were using port 6382 which has changed to 8382 in this release.

Since we deploy an instance of the Coding Tool together with the API, visiting the URL http://localhost:8382/ct should open up the ICD-11 Coding Tool.
Similarly, visiting the URL http://localhost:8382/browse should open up the ICD-11 Browser.
Configuration
By default the API runs at port 8382. You could change it by editing the file appsettings.json at the line that has : "urls": "http://*:8382"
This file should be located at the installation directory; the API is installed at C:\Program Files\ICD-API by default
Please note that for editing the file appsettings.json, you must to open it as an administrator user.

By default the API runs with the latest version of the ICD-11. To change this you may edit the appsettings.json file and change the "include" field.
Here are some examples:

"include": "2024-01_en" means 2024-01 version of ICD-11 in English (default)
"include": "2024-01_tr" means 2024-01 version of ICD-11 in Turkish
"include": "2024-01_ar-en" means 2024-01 version of ICD-11 in Arabic and English
"include": "2023-01_en" means 2023-01 version of ICD-11 in English
Available versions and languages of ICD-11 are listed at the supported classifications. ICD-10 is not supported in this release.

DORIS (Underlying cause of death detection) functionality is in pre-release state. For this reason it is not enabled by default.
You may enable the DORIS endpoints using the "enableDoris": true parameter. For more information about DORIS support see the release notes.

You need to restart the service after making any changes to the appsettings.json file. Windows services could be stopped, started or restarted from the Services app

Updating the ICD-API
To update the ICD-API that is installed as a Windows service, you just need to download the new version of the installation package (.msi file) and run it.

    "api_executable": "C:\\Program Files\\ICD-API\\ICD-API.exe",
    "client_id": "a42ca42f-fe84-4134-839a-78f0d7754cba_aaca6aba-e80b-4fc1-91c5-6ddd0056b906",
    "client_secret": "24GdB5PAgQ0gi3JG6qtx7nxqWjZpK51eEkuuWmj26YM"
}


    "system": {
        "name": "STORME21FERR",
        "manufacturer": "Dell Inc.",
        "model": "OptiPlex 3050",
        "type": "x64-based PC",
        "processor": "Intel(R) Core(TM) i5-7500T CPU @ 2.70GHz, 2712 MHz, 4 Cores, 4 Logical Processors",
        "ram": "32.0 GB",
        "physical_memory": "31.9 GB",
        "available_memory": "18.9 GB",
        "virtual_memory": "36.6 GB",
        "available_virtual_memory": "21.6 GB"
    },
    "storage": [
        {
            "drive": "A:",
            "size": "931.51 GB",
            "free_space": "641.02 GB"
        },
        {
            "drive": "B:",
            "size": "475.92 GB",
            "free_space": "459.46 GB"
        },
        {
            "drive": "C:",
            "size": "1.82 TB",
            "free_space": "1.47 TB"
        },
        {
            "drive": "E:",
            "size": "931.48 GB",
            "free_space": "260.14 GB"
        },
        {
            "drive": "F:",
            "size": "931.51 GB",
            "free_space": "283.46 GB"
        },
        {
            "drive": "G:",
            "size": "465.76 GB",
            "free_space": "464.67 GB"
        },
        {
            "drive": "H:",
            "size": "1.82 TB",
            "free_space": "1.77 TB"
        }
    ]
}
{
    "network": {
        "sharing_and_discovery": {
            "file_and_printer_sharing_service": "Disabled",
            "simple_file_sharing": "Enabled",
            "administrative_shares": "Enabled",
            "network_access_sharing_and_security_model_for_local_accounts": "Classic - local users authenticate as themselves",
            "private_profile": {
                "network_discovery": "Enabled",
                "file_and_printer_sharing": "Disabled"
            },
            "public_profile": {
                "network_discovery": "Disabled",
                "file_and_printer_sharing": "Disabled"
            }
        },
        "adapters_list": {
            "enabled": [
                {
                    "name": "ALOGIC DX3 Ethernet",
                    "connection_name": "Ethernet 2",
                    "dhcp_enabled": "Yes",
                    "mac_address": "00-24-9B-72-47-3C"
                },
                {
                    "name": "Bluetooth Device (Personal Area Network)",
                    "connection_name": "Bluetooth Network Connection",
                    "dhcp_enabled": "Yes",
                    "mac_address": "64-5D-86-4B-37-7C"
                },
                {
                    "name": "Hyper-V Virtual Ethernet Adapter",
                    "connection_name": "vEthernet (WSL)",
                    "netbios_over_tcpip": "Yes",
                    "dhcp_enabled": "No",
                    "mac_address": "00-15-5D-93-A2-DC",
                    "ip_address": "172.20.160.1",
                    "subnet_mask": "255.255.240.0"
                },
                {
                    "name": "Intel(R) Dual Band Wireless-AC 8265",
                    "connection_name": "Wi-Fi",
                    "netbios_over_tcpip": "Yes",
                    "dhcp_enabled": "Yes",
                    "mac_address": "64-5D-86-4B-37-78",
                    "ip_address": "192.168.1.118",
                    "subnet_mask": "255.255.255.0",
                    "gateway_server": "192.168.1.1",
                    "dhcp": "192.168.1.1",
                    "dns_server": [
                        "192.168.1.1",
                        "0.0.0.0"
                    ]
                }
            ]
        }
    }
}
{
    "current_internet_connection": {
        "connected_through": "WireGuard Tunnel",
        "ip_address": "10.2.0.2",
        "subnet_mask": "255.255.255.255",
        "gateway_server": "0.0.0.0",
        "preferred_dns_server": "10.2.0.1",
        "dhcp": "Disabled",
        "adapter_type": "Unknown",
        "netbios_over_tcpip": "Enabled via DHCP",
        "netbios_node_type": "Hybrid node",
        "link_speed": "0 Bps"
    },
    "computer_name": {
        "netbios_name": "STORME21FERR",
        "dns_name": "STORME21FERR",
        "membership": "Part of workgroup",
        "workgroup": "WORKGROUP"
    },
    "remote_desktop": "Disabled",
    "console": {
        "state": "Active",
        "domain": "STORME21FERR"
    },
    "wininet_info": {
        "lan_connection": "Local system uses a local area network to connect to the Internet",
        "ras_connection": "Local system has RAS to connect to the Internet"
    },
    "wifi_info": {
        "using_native_wifi_api_version": 2,
        "available_access_points_count": 4,
        "access_points": [
            {
                "ssid": "Kogan_A308_2.4G",
                "frequency": "2412000 kHz",
                "channel_number": 1,
                "name": "Kogan_A308_2.4G",
                "signal_strength_quality": 31,
                "security": "Enabled",
                "state": "The interface is connected to a network",
                "dot11_type": "Infrastructure BSS network",
                "network_connectible": "Yes",
                "network_flags": "There is a profile for this network",
                "cipher_algorithm": "AES-CCMP algorithm",
                "auth_algorithm": "802.11i RSNA algorithm that uses PSK"
            }
        ]
    }
}
{
    "graphics": {
        "monitors": [
            {
                "name": "MSI MP2412 on Intel HD Graphics 630",
                "current_resolution": "1920x1080 pixels",
                "work_resolution": "1920x1050 pixels",
                "state": "Enabled",
                "multiple_displays": "Extended, Primary, Enabled",
                "width": 1920,
                "height": 1080,
                "bpp": "32 bits per pixel",
                "frequency": "59 Hz",
                "device": "\\\\.\\DISPLAY1\\Monitor0"
            },
            {
                "name": "C27F591 on Intel HD Graphics 630",
                "current_resolution": "1920x1080 pixels",
                "work_resolution": "1920x1050 pixels",
                "state": "Enabled",
                "multiple_displays": "Extended, Secondary, Enabled",
                "width": 1920,
                "height": 1080,
                "bpp": "32 bits per pixel",
                "frequency": "71 Hz",
                "device": "\\\\.\\DISPLAY2\\Monitor0"
            },
            {
                "name": "MSI MP2412 on DisplayLink USB Device",
                "current_resolution": "1920x1080 pixels",
                "work_resolution": "1920x1050 pixels",
                "state": "Enabled",
                "multiple_displays": "Extended, Secondary, Enabled",
                "width": 1920,
                "height": 1080,
                "bpp": "32 bits per pixel",
                "frequency": "60 Hz",
                "device": "\\\\.\\DISPLAY4\\Monitor0"
            },
            {
                "name": "MSI MP2412 on DisplayLink USB Device",
                "current_resolution": "1920x1080 pixels",
                "work_resolution": "1920x1050 pixels",
                "state": "Enabled",
                "multiple displays": "Extended, Secondary, Enabled",
                                "width": 1920,
                "height": 1080,
                "bpp": "32 bits per pixel",
                "frequency": "60 Hz",
                "device": "\\\\.\\DISPLAY5\\Monitor0"
            }
        ],
        "unknown_displaylink_usb_devices": [
            {
                "manufacturer": "Unknown",
                "model": "DisplayLink USB Device",
                "device_id": "17E9-0000",
                "subvendor": "Undefined (0000)",
                "driver_version": "11.4.9747.0"
            },
            {
                "manufacturer": "Unknown",
                "model": "DisplayLink USB Device",
                "device_id": "17E9-0000",
                "subvendor": "Undefined (0000)",
                "driver_version": "11.4.9747.0"
            }
        ],
        "intel_hd_graphics_630": {
            "manufacturer": "Intel",
            "model": "HD Graphics 630",
            "device_id": "8086-5912",
            "revision": "5",
            "subvendor": "Dell (1028)",
            "current_performance_level": "Level 0",
            "driver_version": "31.0.101.2114"
        }
    }
}
{
    "motherboard": {
        "manufacturer": "Dell Inc.",
        "model": "0JP3NX (U3E1)",
        "version": "A01",
        "chipset_vendor": "Intel",
        "chipset_model": "Kaby Lake",
        "chipset_revision": "05",
        "southbridge_vendor": "Intel",
        "southbridge_model": "B250",
        "southbridge_revision": "00",
        "system_temperature": "28 °C",
        "bios": {
            "brand": "Dell Inc.",
            "version": "1.30.0",
            "date": "31/03/2024"
        },
        "pci_data": {
            "slot_pci_e": {
                "slot_type": "PCI-E",
                "slot_usage": "In Use",
                "data_lanes": "x4",
                "slot_designation": "Slot1_M.2",
                "characteristics": "3.3V, PME",
                "slot_number": "0"
            }
        }
    },
    "ram": {
        "memory_slots": {
            "total_memory_slots": 2,
            "used_memory_slots": 2,
            "free_memory_slots": 0
        },
        "memory": {
            "type": "DDR4",
            "size": "32768 MBytes",
            "channels": "Dual",
            "dram_frequency": "1196.8 MHz",
            "cas_latency": "17 clocks",
            "ras_to_cas_delay": "17 clocks",
            "ras_precharge": "17 clocks",
            "cycle_time": "39 clocks",
            "command_rate": "2T"
        }
    }
}
{
    "physical_memory": {
        "memory_usage": "41 %",
        "total_physical": "32 GB",
        "available_physical": "19 GB",
        "total_virtual": "37 GB",
        "available_virtual": "20 GB"
    },
    "spd": [
        {
            "slot": "1",
            "type": "DDR4",
            "size": "16384 MBytes",
            "manufacturer": "SK Hynix",
            "max_bandwidth": "DDR4-2400 (1200 MHz)",
            "part_number": "HMA82GS6AFR8N-UH",
            "serial_number": "328F844C",
            "week_year": "44 / 17"
        },
        {
            "slot": "2",
            "type": "DDR4",
            "size": "16384 MBytes",
            "manufacturer": "SK Hynix",
            "max_bandwidth": "DDR4-2400 (1200 MHz)",
            "part_number": "HMA82GS6AFR8N-UH",
            "serial_number": "328F837A",
            "week_year": "44 / 17"
        }
    ]
}
{
    "cpu": {
        "name": "Intel Core i5 7500T",
        "cores": 4,
        "threads": 4,
        "code_name": "Kaby Lake",
        "package": "Socket 1151 LGA",
        "technology": "14nm",
        "specification": "Intel Core i5-7500T CPU @ 2.70GHz",
        "family": "6",
        "extended_family": "6",
        "model": "E",
        "extended_model": "9E",
        "stepping": "9",
        "revision": "B0",
        "instructions": "MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, Intel 64, NX, AES, AVX, AVX2, FMA3",
        "virtualization": "Not supported",
        "hyperthreading": "Not supported",
        "bus_speed": "99.8 MHz",
        "stock_core_speed": "2700 MHz",
        "average_temperature": "58 °C",
        "caches": {
            "l1_data_cache_size": "4 x 32 KBytes",
            "l1_instructions_cache_size": "4 x 32 KBytes",
            "l2_unified_cache_size": "4 x 256 KBytes",
            "l3_unified_cache_size": "6144 KBytes"
        }
    }
}
     
{
    "cores_info": [
        {
            "core_speed": "3094.7 MHz",
            "multiplier": "x 31.0",
            "bus_speed": "99.8 MHz",
            "temperature": "57 °C",
            "threads": "APIC ID: 0"
        },
        {
            "core_speed": "3094.7 MHz",
            "multiplier": "x 31.0",
            "bus_speed": "99.8 MHz",
            "temperature": "58 °C",
            "threads": "APIC ID: 2"
        },
        {
            "core_speed": "3094.7 MHz",
            "multiplier": "x 31.0",
            "bus_speed": "99.8 MHz",
            "temperature": "58 °C",
            "threads": "APIC ID: 4"
        },
        {
            "core_speed": "3094.7 MHz",
            "multiplier": "x 31.0",
            "bus_speed": "99.8 MHz",
            "temperature": "59 °C",
            "threads": "APIC ID: 6"
        }
    ],
    "machine_variables": {
        "__PSLockDownPolicy": "0",
        "ChocolateyInstall": "C:\\ProgramData\\chocolatey",
        "ComSpec": "C:\\Windows\\system32\\cmd.exe",
        "DriverData": "C:\\Windows\\System32\\Drivers\\DriverData",
        "NUMBER_OF_PROCESSORS": "4",
        "OPENBLAS_ROOT": "C:\\OpenBLAS_0.3.27",
        "OS": "Windows_NT",
        "Path": [
            "C:\\Program Files\\CMake\\bin",
            "C:\\Program Files\\Python38\\Scripts\\",
            "C:\\Program Files\\Python38\\",
            "C:\\Windows\\system32",
            "C:\\Windows",
            "C:\\Windows\\System32\\Wbem",
            "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\",
            "C:\\Windows\\System32\\OpenSSH\\",
            "C:\\Program Files\\Docker\\Docker\\resources\\bin",
            "C:\\Program Files\\PowerShell\\7",
            "C:\\Program Files\\Microsoft VS Code\\bin",
            "C:\\SysinternalsSuite\\",
            "C:\\Program Files\\nodejs",
            "C:\\Program Files\\dotnet\\",
            "C:\\Program Files\\Gpg4win\\bin",
            "C:\\ProgramData\\chocolatey\\bin",
            "C:\\Program Files\\Microsoft SQL Server\\Client SDK\\ODBC\\170\\Tools\\Binn\\",
            "C:\\Program Files\\Microsoft SQL Server\\150\\Tools\\Binn\\",
            "C:\\Users\\Reece\\AppData\\Local\\Microsoft\\WindowsApps\\",
            "C:\\ProgramData\\miniconda3\\condabin",
            "C:\\Program Files\\nodejs\\node_modules\\npm\\bin",
            "C:\\Program Files (x86)\\InteloneAPI\\compiler\\2024.2.0\\bin",
            "C:\\Program Files (x86)\\Windows Kits\\10\\Windows Performance Toolkit\\",
            "C:\\MinGW\\bin",
            "C:\\cygwin64\\bin",
            "C:\\msys64\\usr\\bin",
            "C:\\OSGeo4W\\bin",
            "C:\\msys64\\usr\\bin",
            "C:\\OpenBLAS_0.3.27\\bin",
            "C:\\Program Files (x86)\\Intel\\openvino_2024.3.0",
            "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE",
            "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\Tools",
            "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\MSBuild\\Current\\Bin",
            "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build",
            "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.40.33807\\bin\\Hostx64\\x64",
            "C:\\Program Files\\Python312\\Scripts\\",
            "C:\\Program Files\\Python312\\",
            "C:\\Program Files\\Git\\cmd",
            "C:\\Program Files\\Git\\mingw64\\bin",
            "C:\\Program Files\\Git\\usr\\bin"
        ],
        "PATHEXT": ".COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC;.PY;.PYW",
        "POWERSHELL_DISTRIBUTION_CHANNEL": "MSI:Windows 10 Pro",
        "PROCESSOR_ARCHITECTURE": "AMD64",
        "PROCESSOR_IDENTIFIER": "Intel64 Family 6 Model 158 Stepping 9, GenuineIntel",
        "PROCESSOR_LEVEL": "6",
        "PROCESSOR_REVISION": "9e09",
        "PSModulePath": "%ProgramFiles%\\WindowsPowerShell\\Modules;C:\\Windows\\system32\\WindowsPowerShell\\v1.0\\Modules",
        "TEMP": "C:\\Windows\\TEMP",
        "TMP": "C:\\Windows\\TEMP",
        "USERNAME": "SYSTEM",
        "VS140COMNTOOLS": "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\Common7\\Tools\\",
        "windir": "C:\\Windows",
        "ZES_ENABLE_SYSMAN": "1"
    },
    "user_variables": {
        "USERPROFILE": "C:\\Users\\Reece",
        "SystemRoot": "C:\\Windows",
        "Environment_Variables": {
            "ChocolateyLastPathUpdate": "133666879960947796",
            "HOME": "C:\\Users\\Reece",
            "OneDrive": "C:\\Users\\Reece\\OneDrive",
            "OneDriveConsumer": "C:\\Users\\Reece\\OneDrive",
            "Path": [
                "C:\\msys64\\usr\\bin",
                "C:\\Users\\Reece\\AppData\\Local\\GitHubDesktop\\bin"
            ],
            "TEMP": "C:\\Users\\Reece\\AppData\\Local\\Temp",
            "TMP": "C:\\Users\\Reece\\AppData\\Local\\Temp"
        }
    },
[Python 3.8.8] (venv_openvino) PS E:\Public_Interest_Project\Scripts_Module> pip freeze
abc-classification==0.8
abc-distributions==0.1
abc-utils==0.0.1
abc_xml_converter==1.0.1
about-time==4.2.1
absl-py==1.4.0
accelerate==0.33.0
addict==2.4.0
aiobotocore==2.13.1
aiofiles==24.1.0
aiohappyeyeballs==2.3.4
aiohttp==3.10.0
aiohttp-cors==0.7.0
aioitertools==0.11.0
aiokafka==0.11.0
aiosignal==1.3.1
alabaster==0.7.13
albucore==0.0.13
albumentations==1.0.3
alive-progress==3.1.5
annotated-types==0.7.0
anomalib==0.7.0
ansicon==1.89.0
antlr4-python3-runtime==4.9.3
anyio==4.4.0
appdirs==1.4.4
arabic-reshaper==3.0.0
argon2-cffi==23.1.0
argon2-cffi-bindings==21.2.0
array-record==0.4.0
arrow==1.3.0
asgiref==3.8.1
asn1crypto==1.5.1
astor==0.8.1
asttokens==2.4.1
astunparse==1.6.3
async-lru==2.0.4
async-timeout==4.0.3
attrs==23.1.0
audioread==3.0.1
autograd==1.6.2
av==12.3.0
Babel==2.15.0
backcall==0.2.0
backports-abc==0.5
backports.zoneinfo==0.2.1
bandit==1.7.9
beautifulsoup4==4.12.3
bleach==6.1.0
blessed==1.20.0
blinker==1.8.2
blis==0.7.11
blobconverter==1.4.3
boto3==1.34.151
botocore==1.34.152
bottle==0.12.25
Brotli==1.1.0
bs4==0.0.2
build==1.2.1
cachetools==5.4.0
catalogue==2.0.10
certifi==2024.7.4
cffi==1.16.0
chardet==3.0.4
charset-normalizer==3.3.2
click==8.1.7
cloudevents==1.11.0
cloudpathlib==0.18.1
cma==3.2.2
colorama==0.4.6
coloredlogs==15.0.1
colorful==0.5.6
Columnar==1.4.1
comm==0.2.2
common==0.1.2
confection==0.1.5
contourpy==1.1.1
controlnet-aux==0.0.9
coverage==5.3.1
cryptography==43.0.0
cssselect==1.2.0
cssselect2==0.7.0
cuda-python==12.3.0
cycler==0.12.1
cymem==2.0.8
Cython==0.29.30
data==0.4
data-gradients==0.3.2
data-platform==0.0.1
dataclasses-json==0.6.7
datasets==2.20.0
datumaro==1.5.2
ddg==0.2.2
debugpy==1.8.2
decorator==5.1.1
deeplite-torch-zoo==2.0.5
deepvoice3-pytorch @ git+https://github.com/hash2430/dv3_world@4740bfa06fade18b2288925c5b14b71f547e4f73
defusedxml==0.7.1
Deprecated==1.2.14
deprecation==2.1.0
diffusers==0.29.2
dill==0.3.8
dirtyjson==1.0.8
distlib==0.3.8
distro==1.9.0
dm-tree==0.1.8
dnspython==2.6.1
docker==4.4.4
docopt==0.6.2
docstring_parser==0.16
docutils==0.17.1
dual==0.0.10
duckduckgo_search==6.2.5
dynamo3==0.4.10
editor==1.6.6
einops==0.3.2
emit==0.4.0
etils==1.3.0
eval_type_backport==0.2.0
exceptiongroup==1.2.2
executing==2.0.1
fake-useragent==1.5.1
fastapi==0.109.2
fastavro==1.9.5
fastdtw==0.3.4
fastjsonschema==2.17.1
filelock==3.15.4
filetype==1.2.0
fire==0.6.0
Flask==3.0.3
flatbuffers==24.3.25
flywheel==0.5.4
fonttools==4.53.1
fpdf==1.7.2
fqdn==1.5.1
FrEIA==0.2
frozenlist==1.4.1
fsspec==2024.6.1
ftfy==6.2.0
funcsigs==1.0.2
future==1.0.0
gast==0.4.0
gcrypter==0.4
gcsfs==2024.6.1
geti-sdk==1.16.1
gevent==24.2.1
geventhttpclient==2.0.2
git-client==0.2.3
gitdb==4.0.11
GitPython==3.1.43
google-api-core==2.19.1
google-api-python-client==2.139.0
google-auth==2.32.0
google-auth-httplib2==0.2.0
google-auth-oauthlib==1.0.0
google-cloud-core==2.4.1
google-cloud-storage==2.18.0
google-crc32c==1.5.0
google-pasta==0.2.0
google-resumable-media==2.7.1
googleapis-common-protos==1.63.2
grapheme==0.6.0
graphviz==0.20.3
greenlet==3.0.3
grpcio==1.65.2
h11==0.14.0
h5py==3.10.0
hparams==0.3.0
html5lib==1.1
httpcore==1.0.5
httplib2==0.22.0
httptools==0.6.1
httpx==0.26.0
huggingface-hub==0.23.5
humanfriendly==10.0
hydra-core==1.3.2
ibm-cloud-sdk-core==3.20.4
ibm-watson==8.1.0
idna==2.8
ijson==3.3.0
imagecodecs==2023.3.16
imagededup==0.3.2
imageio==2.34.2
imageio-ffmpeg==0.4.8
imagesize==1.4.1
imbalanced-learn==0.12.3
imblearn==0.0
imgaug==0.4.0
importlib==1.0.4
importlib-resources==5.13.0
importlib_metadata==8.0.0
infbench==0.0.1
inflect==7.3.1
iniconfig==2.0.0
inquirer==3.3.0
intervaltree==3.1.0
ipykernel==6.29.5
ipython==8.12.3
ipywidgets==8.1.3
isoduration==20.11.0
itsdangerous==2.2.0
jaconv==0.4.0
jax==0.4.13
jedi==0.19.1
Jinja2==3.1.4
jinxed==1.3.0
jmespath==1.0.1
joblib==1.4.2
jp==0.2.4
json-stream==2.3.2
json-stream-rs-tokenizer==0.4.26
json-tricks==3.16.1
json5==0.9.25
jsonargparse==4.32.0
jsonpointer==3.0.0
jsonschema==3.2.0
jsonschema-specifications==2023.12.1
jstyleson==0.0.2
jupyter==1.0.0
jupyter-console==6.6.3
jupyter-events==0.10.0
jupyter-lsp==2.2.5
jupyter_client==8.6.2
jupyter_core==5.7.2
jupyter_server==2.14.2
jupyter_server_terminals==0.5.3
jupyterlab==4.2.4
jupyterlab_pygments==0.3.0
jupyterlab_server==2.27.3
jupyterlab_widgets==3.0.11
kecam==1.4.1
keras==2.13.1
keras-cv-attention-models==1.4.1
kiwisolver==1.4.5
kornia==0.6.9
kserve==0.13.1
kubernetes==30.1.0
label-studio-sdk==1.0.4
langcodes==3.4.0
langdetect==1.0.9
language_data==1.2.0
lazy_loader==0.4
libclang==18.1.1
librosa==0.10.2.post1
lightning-utilities==0.11.6
llama-index-core==0.10.59
llama-index-postprocessor-openvino-rerank==0.1.6
llvmlite==0.41.1
lmdb==1.5.1
loguru==0.7.2
luxonis-ml==0.2.3
lws==1.2.8
lxml==5.2.2
marisa-trie==1.2.0
Markdown==3.6
markdown-it-py==3.0.0
MarkupSafe==2.1.5
marshmallow==3.21.3
matplotlib==3.7.5
matplotlib-inline==0.1.7
mdurl==0.1.2
mecab-python3==1.0.9
mediapipe==0.10.11
mistune==3.0.2
ml-dtypes==0.2.0
mlserver==1.4.0
mlserver-openvino==0.4.10
model-index==0.1.11
modelconv==0.1.2
more-itertools==10.3.0
mpmath==1.3.0
msgpack==1.0.8
multidict==6.0.5
multiprocess==0.70.16
murmurhash==1.0.10
mxnet==1.2.0
mypy-extensions==1.0.0
natsort==8.4.0
nbclient==0.10.0
nbconvert==7.16.4
nbformat==5.10.4
nbsphinx==0.9.4
nest-asyncio==1.6.0
netron==7.8.1
networkx==2.8
nibabel==5.2.1
ninja==1.11.1.1
nltk==3.8.1
nncf==2.12.0
nnmnkwii==0.1.3
nose==1.3.7
notebook==7.2.1
notebook_shim==0.2.4
numba==0.58.1
numpy==1.23.4
oauth2client==4.1.3
oauthlib==3.2.2
omegaconf==2.3.0
onnx==1.13.0
onnx-simplifier==0.4.36
onnxruntime==1.13.1
onnxsim==0.4.36
openai==1.37.1
opencensus==0.11.4
opencensus-context==0.1.3
opencv-contrib-python==4.10.0.84
opencv-python==4.9.0.80
opencv-python-headless==4.10.0.84
opencv-python-inference-engine==4.0.1.2
openmim==0.3.7
opentelemetry-api==1.26.0
opentelemetry-exporter-otlp-proto-common==1.26.0
opentelemetry-exporter-otlp-proto-grpc==1.26.0
opentelemetry-instrumentation==0.47b0
opentelemetry-instrumentation-asgi==0.47b0
opentelemetry-instrumentation-fastapi==0.47b0
opentelemetry-instrumentation-grpc==0.47b0
opentelemetry-proto==1.26.0
opentelemetry-sdk==1.26.0
opentelemetry-semantic-conventions==0.47b0
opentelemetry-util-http==0.47b0
openvino==2023.0.0
openvino-dev==2023.0.0
openvino-genai==2024.3.0.0
openvino-model-api==0.1.5
openvino-telemetry==2024.1.0
openvino-tokenizers==2024.3.0.0
openvino-trackface==0.0.1
openvino-workbench==2022.3.0
openvino2tensorflow==1.34.0
openvino_kaggle==1.0.0
opt-einsum==3.3.0
optimize_tensorrt==1.0.1
optimum==1.20.0
optimum-intel==1.18.0
ordered-set==4.1.0
orjson==3.9.15
oscrypto==1.3.0
otx==1.4.4
ouroboros_hf_text_generation==1.0.0
ovcontrolnet-tools==1.0.2
overrides==7.7.0
ovmsclient==2023.1
packaging==24.1
paddle==1.0.2
paddleocr==2.8.1
paddlepaddle==2.6.1
pandas==2.0.3
pandoc==2.3
pandocfilters==1.5.1
parse==1.20.2
parso==0.8.4
pathlib==1.0.1
pathlib_abc==0.3.1
pathvalidate==3.2.0
pbr==6.0.0
pdf2image==1.17.0
peewee==3.17.6
pickleshare==0.7.5
pillow==10.3.0
pip-tools==7.4.1
pkgutil_resolve_name==1.3.10
platformdirs==4.2.2
plotly==5.23.0
pluggy==1.5.0
plumbum==1.8.3
ply==3.11
polars==1.3.0
polygraphy==0.49.9
polygraphy-trtexec==0.0.9
pooch==1.8.2
preshed==3.0.9
primp==0.5.5
prometheus_client==0.20.0
promise==2.3
prompt_toolkit==3.0.47
proto-plus==1.24.0
protobuf==4.25.4
prox==0.0.17
psutil==5.9.0
pure_eval==0.2.3
py==1.11.0
py-cpuinfo==9.0.0
py-grpc-prometheus==0.8.0
py-spy==0.3.14
py2gc==1.0.6
pyarrow==17.0.0
pyarrow-hotfix==0.6
pyasn1==0.6.0
pyasn1_modules==0.4.0
pyclipper==1.3.0.post5
pycocotools==2.0.6
pycocotools-windows==2.0.0.2
pycparser==2.22
pydantic==2.8.2
pydantic-settings==2.4.0
pydantic_core==2.20.1
pyDeprecate==0.3.2
pydot==2.0.0
pydub==0.25.1
pyee==11.1.0
pyemd==1.0.0
Pygments==2.18.0
pyHanko==0.25.1
pyhanko-certvalidator==0.26.3
PyJWT==2.9.0
pymongo==4.5.0
pymoo==0.6.1.2
PyMuPDF==1.24.9
PyMuPDFb==1.24.9
pypandoc==1.13
pyparsing==2.4.5
pypdf==4.3.1
PyPDF2==3.0.1
pypng==0.20220715.0
pyppeteer==2.0.0
pyproject_hooks==1.1.0
pyquery==2.0.0
pyreadline3==3.4.1
pyrsistent==0.20.0
PySocks==1.7.1
pysptk==1.0.1
pytesseract==0.3.10
pytest==8.3.2
python-bidi==0.6.0
python-dateutil==2.9.0.post0
python-docx==1.1.2
python-dotenv==1.0.1
python-geoip-python3==1.3
python-json-logger==2.0.7
python-multipart==0.0.9
python-rapidjson==1.19
pytorch-lightning==1.9.5
pytorchcv==0.0.67
pytz==2024.1
pyvww==0.1.1
PyWavelets==1.4.1
pywin32==306
pywinpty==2.0.13
PyYAML==6.0.1
pyzmq==26.0.3
qrcode==7.4.2
qtconsole==5.5.2
QtPy==2.4.1
qudida==0.0.4
rapid-layout==0.3.0
rapidfuzz==3.9.5
rapidocr-api==0.0.7
rapidocr-onnxruntime==1.3.24
rapidocr-openvino==1.3.24
rapidocr-openvinogpu==0.0.9
rapidocr-pdf==0.1.0
rapidocr-web==0.1.10
ratelimit==2.2.1
ray==2.10.0
readchar==4.1.0
referencing==0.35.1
regex==2024.7.24
reportlab==3.6.13
requests==2.31.0
requests-html==0.10.0
requests-mock==1.12.1
requests-oauthlib==2.0.0
requests-toolbelt==1.0.0
retry==0.9.2
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rich==13.7.1
rpds-py==0.19.1
rsa==4.9
ruamel.yaml==0.18.6
ruamel.yaml.clib==0.2.8
runs==1.2.2
s3fs==2024.6.1
s3transfer==0.10.2
safetensors==0.4.3
scikit-image==0.21.0
scikit-learn==1.3.2
scipy==1.10.1
seaborn==0.13.2
semver==3.0.2
Send2Trash==1.8.3
sentencepiece==0.2.0
Shapely==1.8.0
shellingham==1.5.4
simplejson==3.19.2
simsimd==4.4.0
six==1.16.0
smart-open==7.0.4
smmap==5.0.1
sniffio==1.3.1
snowballstemmer==2.2.0
sortedcontainers==2.4.0
sounddevice==0.4.7
soundfile==0.12.1
soupsieve==2.5
soxr==0.3.7
spacy==3.7.5
spacy-legacy==3.0.12
spacy-loggers==1.0.5
Sphinx==4.0.2
sphinx-autodoc-typehints==2.0.1
sphinx-rtd-theme==1.3.0
sphinxcontrib-applehelp==1.0.4
sphinxcontrib-devhelp==1.0.2
sphinxcontrib-htmlhelp==2.0.1
sphinxcontrib-jquery==4.1
sphinxcontrib-jsmath==1.0.1
sphinxcontrib-qthelp==1.0.3
sphinxcontrib-serializinghtml==1.1.5
SQLAlchemy==2.0.31
srsly==2.4.8
stack-data==0.6.3
starlette==0.36.3
starlette_exporter==0.23.0
stevedore==5.2.0
string-color==1.2.3
stringcase==1.2.0
super-gradients==3.3.0
svglib==1.5.1
sympy==1.13.1
synthesisai==0.5.2
tabulate==0.9.0
tag==0.5
tenacity==8.5.0
tensorboard==2.13.0
tensorboard-data-server==0.7.2
tensorboardX==2.4.1
tensorflow==2.13.0
tensorflow-datasets==4.9.2
tensorflow-estimator==2.13.0
tensorflow-intel==2.13.0
tensorflow-io-gcs-filesystem==0.31.0
tensorflow-metadata==1.14.0
termcolor==1.1.0
terminado==0.18.1
tesseract==0.1.3
textblob==0.18.0.post0
texttable==1.6.4
tflite2tensorflow==1.22.0
Theano==1.0.5
thinc==8.2.5
thop==0.1.1.post2209072238
threadpoolctl==3.5.0
tifffile==2023.7.10
tiffile==2018.10.18
tight==0.1.0
tiktoken==0.7.0
timing-asgi==0.3.1
timm==0.9.2
tinycss2==1.3.0
tokenizers==0.19.1
toml==0.10.2
tomli==2.0.1
toolz==0.12.1
torch==2.0.1
torchaudio==2.4.0
torchmetrics==0.8.0
torchprofile==0.0.4
torchvision==0.15.2
tornado==6.4.1
tqdm==4.65.2
train==0.0.5
traitlets==5.14.3
transformers==4.41.2
treelib==1.6.1
tritonclient==2.41.1
typeguard==4.3.0
typer==0.12.3
types-python-dateutil==2.9.0.20240316
typeshed_client==2.7.0
typing==3.7.4.3
typing-inspect==0.9.0
typing_extensions==4.12.2
tzdata==2024.1
tzlocal==5.2
uform==3.0.2
ujson==5.10.0
ultralytics==8.0.200
Unidecode==1.3.8
uri-template==1.3.0
uritemplate==4.1.1
uritools==4.0.3
urllib3==1.26.19
utils==1.0.2
uvicorn==0.21.1
vector_forge==0.0.1
virtualenv==20.26.3
w3lib==2.2.1
Wand==0.6.11
wasabi==1.1.3
watchfiles==0.22.0
wcwidth==0.2.13
weasel==0.4.1
webcolors==24.6.0
webencodings==0.5.1
websocket-client==1.8.0
websockets==10.4
Werkzeug==3.0.3
widgetsnbextension==4.0.11
win32-setctime==1.1.0
wrapt==1.14.1
xhtml2pdf==0.2.11
xmljson==0.2.1
xmod==1.8.1
xxhash==3.4.1
yarl==1.9.4
zipp==3.19.2
zope.event==5.0
zope.interface==6.4.post2

(Venv_Python_3_9) PS E:\Public_Interest_Project\Venv_Python_3_9> pip freeze
a-cv-imwrite-imread-plus==0.13
about-time==4.2.1
alive-progress==3.1.5
asteval==1.0.2
astropy==6.0.1
astropy-iers-data==0.2024.7.29.0.32.7
astroquery==0.4.7
autograd==1.6.2
backports.tarfile==1.2.0
beautifulsoup4==4.12.3
bokeh==3.4.3
callpyfile==0.10
camelot-py==0.11.0
certifi==2024.7.4
cffi==1.16.0
chardet==5.2.0
charset-normalizer==3.3.2
click==8.1.7
cma==3.2.2
colorama==0.4.6
concurrent-progressbar==1.1.3
contourpy==1.2.1
cryptography==43.0.0
cycler==0.12.1
Deprecated==1.2.14
dill==0.3.8
distro==1.9.0
emcee==3.1.6
et-xmlfile==1.1.0
exceptiongroup==1.2.2
fonttools==4.53.1
future==1.0.0
fuzzywuzzy==0.18.0
ghostscript==0.7
grapheme==0.6.0
html5lib==1.1
idna==3.7
imageio==2.34.2
importlib-metadata==8.2.0
importlib-resources==6.4.0
iniconfig==2.0.0
isiter==0.10
iso639==0.1.4
jaraco.classes==3.4.0
jaraco.context==5.3.0
jaraco.functools==4.0.2
jinja2==3.1.4
joblib==1.4.2
kepler.py==0.0.7
keyring==25.3.0
kiwisolver==1.4.5
langdetect==1.0.9
lazy-loader==0.4
Levenshtein==0.25.1
lmfit==1.3.2
MarkupSafe==2.1.5
matplotlib==3.9.1
more-itertools==10.3.0
multiprocess==0.70.16
networkx==3.2.1
numpy==2.0.1
opencv-python==4.10.0.84
openpyxl==3.1.5
packaging==24.1
pandas==2.2.2
pathos==0.3.2
pdf2image==1.17.0
pdfminer.six==20240706
PDFScraper==1.1.9
pdLSR==0.3.6
pillow==10.4.0
pluggy==1.5.0
pox==0.3.4
ppft==1.7.6.8
psutil==6.0.0
pycparser==2.22
pyerfa==2.0.1.4
pyGTC==0.5.0
pyLIMA==1.9.6
pymoo==0.6.1.3
pyparsing==3.1.2
pypdf==4.3.1
pytesseract==0.3.10
pytest==8.3.2
pytest-astropy-header==0.2.2
python-dateutil==2.9.0.post0
python-Levenshtein==0.25.1
pytz==2024.1
pyvo==1.5.2
pywin32-ctypes==0.2.2
PyYAML==6.0.1
rapidfuzz==3.9.5
requests==2.32.3
scikit-image==0.24.0
scikit-learn==1.5.1
scipy==1.13.1
six==1.16.0
soupsieve==2.5
speclite==0.19
tabula-py==2.9.3
tabulate==0.9.0
tesseractmultiprocessing==0.10
threadpoolctl==3.5.0
tifffile==2024.7.24
tolerant-isinstance==0.10
tomli==2.0.1
tornado==6.4.1
touchtouch==0.11
tqdm==4.66.4
typing-extensions==4.12.2
tzdata==2024.1
uncertainties==3.2.2
urllib3==2.2.2
VBBinaryLensing==3.7.0
Wand==0.6.13
webencodings==0.5.1
wrapt==1.16.0
xyzservices==2024.6.0
yattag==1.15.2
zipp==3.19.2

(venv_python_3_12) PS E:\Public_Interest_Project\venv_python_3_12> pip freeze
appdirs==1.4.4
beautifulsoup4==4.12.3
bs4==0.0.2
certifi==2024.7.4
charset-normalizer==3.3.2
colorama==0.4.6
cssselect==1.2.0
fake-useragent==1.5.1
idna==3.7
importlib_metadata==8.2.0
kiwisolver==1.4.5
lxml==5.2.2
mpmath==1.3.0
parse==1.20.2
pyee==11.1.0
pyppeteer==2.0.0
pyquery==2.0.0
pyrca==0.2.2
requests==2.32.3
requests-html==0.10.0
soupsieve==2.5
tqdm==4.66.4
typing==3.7.4.3
typing_extensions==4.12.2
urllib3==1.26.19
w3lib==2.2.1
websockets==10.4
zipp==3.19.2import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
try:
    # [Code block]
except SpecificException as e:
    logging.error(f'Specific error occurred: {e}')
except Exception as e:
    logging.error(f'Unexpected error occurred: {e}')
    raise
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def log_debug_info(info):
    logging.debug(f'Debug info: {info}')
# Example of integrating a new feature
def new_feature():
    print("This is a new feature")
# Example of refining an existing feature
def refined_feature():
    print("This is a refined feature")
# Implementing advanced data extraction techniques
def extract_data(file_path):
    # Placeholder for data extraction logic
    pass
# Example of optimizing code
def optimized_function():
    # Placeholder for optimized code
    pass
# Implementing automated report generation
def generate_report(data):
    # Placeholder for report generation logic
    pass
# Implementing validation and testing
def validate_test():
    # Placeholder for validation and testing logic
    pass
# Finalizing documentation
def document():
    # Placeholder for documentation logic
    pass
# Implementing deployment and monitoring
def deploy_monitor():
    # Placeholder for deployment and monitoring logic
    pass
# Implementing review and handoff
def review_handoff():
    # Placeholder for review and handoff logic
    pass
