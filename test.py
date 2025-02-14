# test.py
import orbslam3

# Provide paths to the vocabulary and settings files.
# For a minimal test, you can use dummy paths if you are only verifying the binding.
voc_file = "path/to/vocabulary.txt"
settings_file = "path/to/settings.yaml"

# Call the minimal SLAM function.
result = orbslam3.run_orb_slam3(voc_file, settings_file)
print(result)
