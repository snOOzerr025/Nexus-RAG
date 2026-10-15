#!/bin/bash
# Auto-commit script for RAG pipeline project
# Commits files according to manifest every minute, pushes every 5 minutes

# Configuration
MANIFEST_FILE="commit_manifest.json"
STATE_FILE=".auto_commit_state"
INTERVAL_MINUTES=1  # Commit interval
PUSH_EVERY_N_COMMITS=5  # Push every 5 commits (5 minutes at 1 min/commit)

# Ensure we're in the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Initialize state file if it doesn't exist
if [[ ! -f "$STATE_FILE" ]]; then
    echo '{"last_commit_index": -1, "last_push_index": -1}' > "$STATE_FILE"
fi

# Load current state
state=$(cat "$STATE_FILE")
last_commit_index=$(echo "$state" | grep -o '"last_commit_index": [0-9]*' | cut -d' ' -f2)
last_push_index=$(echo "$state" | grep -o '"last_push_index": [0-9]*' | cut -d' ' -f2)

# Get total commits
total_commits=$(jq '.total_commits' "$MANIFEST_FILE")

# Check if we're done
if [[ $last_commit_index -ge $((total_commits - 1)) ]]; then
    echo "All commits completed! Finalizing..."
    git push origin main
    echo "Project ready for production!"
    exit 0
fi

# Calculate next commit index
next_index=$((last_commit_index + 1))

# Get commit details from manifest
commit_info=$(jq -r ".commit_sequence[$next_index]" "$MANIFEST_FILE")
files_array=$(echo "$commit_info" | jq -r '.files[]')
description=$(echo "$commit_info" | jq -r '.description')

# Convert newline-separated files to space-separated
files=()
while IFS= read -r file; do
    if [[ -n "$file" && -f "$file" ]]; then
        files+=("$file")
    fi
done <<< "$files_array"

# Commit if we have files to commit
if [[ ${#files[@]} -gt 0 ]]; then
    # Add files to staging
    git add "${files[@]}"

    # Commit with descriptive message
    commit_message="feat: $description"
    git commit -m "$commit_message"

    # Update state
    new_state=$(echo "$state" | jq --argjson idx "$next_index" '.last_commit_index = $idx')
    echo "$new_state" > "$STATE_FILE"

    echo "$(date): Committed $description (${#files[@]} files)"
else
    # Skip if no files
    echo "$(date): No files to commit for index $next_index, skipping..."
    # Still advance index to avoid infinite loop
    new_state=$(echo "$state" | jq --argjson idx "$next_index" '.last_commit_index = $idx')
    echo "$new_state" > "$STATE_FILE"
fi

# Check if we need to push
if [[ $((next_index % PUSH_EVERY_N_COMMITS)) -eq 0 && $last_push_index -lt $next_index ]]; then
    git push origin main
    new_state=$(echo "$state" | jq --argjson idx "$next_index" '.last_push_index = $idx')
    echo "$new_state" > "$STATE_FILE"
    echo "$(date): Pushed to remote after commit $next_index"
fi

# Schedule next run
echo "Next commit in $INTERVAL_MINUTES minute(s)..."