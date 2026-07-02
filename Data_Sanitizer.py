class DataSanitizer:
    def __init__(self, measurements): #"measurements" is a dictionary and contains {"Height":, "Chest":, "Waist":, "Hip":, "Arm Length":}
        # .copy() ensures we don't accidentally mutate the original data source
        self.measurements = measurements.copy()
        self.errors = [] # A list to collect all errors within the input
        
        self.unit_correction()
        self.proportion_validation()
        self.fallback_data()
    
    def unit_correction(self):
        # .get() is safer. It returns None if the key doesn't exist, preventing KeyErrors.
        height = self.measurements.get("Height") 
        
        if height and 55 < height <= 100: #tallest person in the world height and minimum average height in india in inches
            # If height is in inches, convert ALL non-empty measurements to cm
            for key, value in self.measurements.items():
                if value is not None:
                    self.measurements[key] = round(value * 2.54, 2)
    
    def proportion_validation(self): #using Height as main point of reference because most people tend to know their height as compared to other measurements
        height = self.measurements.get("Height")
        chest = self.measurements.get("Chest")
        waist = self.measurements.get("Waist")
        
        # We must verify the data exists before doing math on it
        if height and chest:
            if chest < (0.3 * height) or chest > (0.55 * height):
                self.errors.append("Chest measurement " + str(chest) + "cm is an outlier.")
                
        if height and waist:
            if waist > (0.6 * height) or waist < (0.35 * height):
                self.errors.append("Waist measurement " + str(waist) + "cm cannot be larger than height.")
    
    def fallback_data(self):
        arm_length = self.measurements.get("Arm Length")
        height = self.measurements.get("Height")
        
        # If height exists, and arm length is either missing entirely, None, or 0
        if height and (arm_length is None or arm_length == 0):
            # Arm Length is approx 44% of total height
            self.measurements["Arm Length"] = round(height * 0.44, 2)

# --- Testing the Engine ---
mock_user_data = {
    "Height": 71, 
    "Chest": 105, 
    "Waist": 180, 
    "Hip": 100, 
    "Arm Length": None
}

sanitizer = DataSanitizer(mock_user_data)
print("Sanitized Data:", sanitizer.measurements)
print("Errors Found:", sanitizer.errors)