#!/bin/bash

bash scripts/LongForecasting/HDKAN.sh
bash scripts/ShortForecasting/HDKAN_s1.sh
bash scripts/ShortForecasting/HDKAN_s2.sh

echo "All sh files finished."
read -p "Press Enter to exit..."
