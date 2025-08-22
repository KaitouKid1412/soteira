#!/bin/bash

# This script clears the contents of specified folders within the events directory.

# Clear all files in the llm_analysis directory
echo "Clearing llm_analysis directory..."
rm -rf /Users/swapnilbajpai/work/hackathon/soteira-2/events/llm_analysis/*

# Clear all files in the llm_sent directory
echo "Clearing llm_sent directory..."
rm -rf /Users/swapnilbajpai/work/hackathon/soteira-2/events/llm_sent/*

# Remove all .jpg files from the main events directory
echo "Removing .jpg files from events directory..."
find /Users/swapnilbajpai/work/hackathon/soteira-2/events/ -maxdepth 1 -type f -name "*.jpg" -delete

echo "Removing .csv files from events directory..."
find /Users/swapnilbajpai/work/hackathon/soteira-2/events/ -maxdepth 1 -type f -name "*.csv" -delete

echo "Removing .json files from events directory..."
find /Users/swapnilbajpai/work/hackathon/soteira-2/events/ -maxdepth 1 -type f -name "*.json" -delete



echo "Cleanup complete."