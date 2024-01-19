  // JavaScript for updating progress bar and handling form elements goes here
    $('#submit-button').on('click', function() {
        // Example: animate the progress bar to 100% over 3 seconds
        $('#progress-bar').css('width', '100%').attr('aria-valuenow', 100).text('Processing...');
        });
        $(document).ready(function() {
        toggleSelectAll(true);
    });
 // Initialize arrays to store data for the selected channels
    var combinedData = [];
    var globalWindowLength;
    var globalWindowOverlap;
    var globalSelectedData = {};
    var globalZoomedChannelData = {};
    var globalSelectedFeatures = {};


    // Define the layout for the combined plot
    var combinedLayout = {
        title: 'Data Plot',
        xaxis: { title: 'Simple' },
        yaxis: { title: 'Amplitude' },
    };

    // Function to update the plot
    function updatePlot(newData) {
        console.log(newData)
        Plotly.react('combined-plot', newData, combinedLayout);
        globalZoomedChannelData = newData.map(function(trace) {
            return {
                name: trace.name,               // channel name from newData
                data: trace.y                   // the y-values from newData
            };
        });
    }
     function updatePlot1(newData) {
        Plotly.react('combined-plot-1', newData, combinedLayout);
    }

    // Initial plot
    updatePlot(combinedData);

    var  areAllChannelsSelected = false;

   document.addEventListener('DOMContentLoaded', function() {

    // Assuming you have a function to update the plot
    var updatedData = combinedData.map(function(trace) {
        return {...trace, visible: true};  // Set each trace to be fully visible
    });

    updatePlot(updatedData);
});
document.getElementById('select-all').addEventListener('click', function() {
        areAllChannelsSelected = true;
        var updatedData = combinedData.map(function(trace) {
            return {...trace, visible: true};  // Set trace to be fully visible
        });

        updatePlot(updatedData);
    });

    // Deselect All Channels
    document.getElementById('deselect-all').addEventListener('click', function() {
        var updatedData = combinedData.map(function(trace) {
            return {...trace, visible: 'legendonly'};  // Set trace to be visible in legend only
        });

        updatePlot(updatedData);
    });

    document.getElementById('combined-plot').on('plotly_relayout', function(eventData) {
    // Check if the event data includes x-axis range information
    if (eventData['xaxis.range[0]'] && eventData['xaxis.range[1]']) {
        // Retrieve the start and end of the zoomed range
        var zoomStart = eventData['xaxis.range[0]'];
        var zoomEnd = eventData['xaxis.range[1]'];

        // Get the Plotly graph div element
        var plotDiv = document.getElementById('combined-plot');

        // Get the data for all traces (channels)
        var allTracesData = plotDiv.data;

        // Filter for visible traces
        var visibleTracesData = allTracesData.filter(function(trace) {
            return trace.visible === true;
        });

              var zoomedChannelData = visibleTracesData.map(function(trace) {
                // Find the index range for the zoomed data
                var startIndex = trace.x.findIndex(x => x >= zoomStart);
                var endIndex = trace.x.findIndex(x => x > zoomEnd);

                // Extract the y-values based on the start and end index
                var dataWithinZoomRange = trace.y.slice(startIndex, endIndex === -1 ? undefined : endIndex + 1);

                return {
                    name: trace.name,               // channel name
                    data: dataWithinZoomRange  // Only the y-values, Entire Selected channel data
                };
            });

            // Check if we got data after filtering
            if (zoomedChannelData.length === 0) {
                console.log('No data found for selected channels.');
            } else {
                // Now you have the zoomed data for each visible channel
                // You can process this data as needed
              //alert(JSON.stringify(zoomedChannelData));
              globalZoomedChannelData = zoomedChannelData;
            }

        // Map through the visible traces and get the zoomed range data

    }
});

function sanitizeId(id) {
    return id.replace(/\s+/g, '').replace(/[^\w-]+/g, '');
}

function createSelectAllCheckbox(category) {
    var sanitizedCategory = sanitizeId(category);
    var selectAllDiv = document.createElement('div');
    selectAllDiv.className = 'form-check';

    var selectAllCheckbox = document.createElement('input');
    selectAllCheckbox.type = 'checkbox';
   selectAllCheckbox.className = 'form-check-input select-all-checkbox';
    selectAllCheckbox.id = sanitizedCategory + '-select-all';

    var selectAllLabel = document.createElement('label');
    selectAllLabel.className = 'form-check-label';
    selectAllLabel.htmlFor = selectAllCheckbox.id;
    selectAllLabel.textContent = 'Select All';

    selectAllDiv.appendChild(selectAllCheckbox);
    selectAllDiv.appendChild(selectAllLabel);

    selectAllCheckbox.addEventListener('change', function() {
        var allCheckboxes = document.querySelectorAll('input[type="checkbox"][data-category="' + sanitizedCategory + '"]');
        allCheckboxes.forEach(function(checkbox) {
            checkbox.checked = selectAllCheckbox.checked;
        });
    });

    return selectAllDiv;
}

function generateCheckboxes() {
    var data_categories = {
        'Temporal': ['Fundamental frequency', 'Maximum peak', 'Roll off', 'Roll on'],
        'Statistical': ['Mean', 'Median', 'Standard Deviation', 'Variance', 'Kurtosis', 'Skewness', 'Interquartile Range', 'Mean absolute deviation', 'Median absolute difference', 'Mean absolute difference'],
        'Signal Processing/Spectral': ['Spectral distance', 'Spectral kurtosis', 'Spectral skewness', 'Spectral spread', 'Cumulative centroid', 'Centroid', 'Max frequency'],
        'Energy-Related': ['Total energy', 'Absolute energy', 'Area under the curve', 'Root mean square'],
        'Other': ['Entropy', 'Maximum', 'Minimum', 'Distance']
    };

    var listContainer = document.getElementById('featuresList');
    if (!listContainer) {
        console.error("The container for the features list was not found.");
        return;
    }

    listContainer.innerHTML = ''; // Clear existing content

    Object.keys(data_categories).forEach(function(category) {
        var categoryDiv = document.createElement('div');
        categoryDiv.className = 'category-div mb-3';

        var categoryHeader = document.createElement('h5');
        categoryHeader.textContent = category;
        categoryDiv.appendChild(categoryHeader);

        var selectAllDiv = createSelectAllCheckbox(category);
        categoryDiv.appendChild(selectAllDiv);

        data_categories[category].forEach(function(feature, index) {
            var featureDiv = document.createElement('div');
            featureDiv.className = 'form-check';

            var featureCheckbox = document.createElement('input');
            featureCheckbox.type = 'checkbox';
            featureCheckbox.className = 'form-check-input';
            featureCheckbox.id = sanitizeId(category) + '-' + index;
            featureCheckbox.value = feature; // Set the value attribute to the feature name
            featureCheckbox.dataset.category = sanitizeId(category);

            var featureLabel = document.createElement('label');
            featureLabel.className = 'form-check-label';
            featureLabel.htmlFor = featureCheckbox.id;
            featureLabel.textContent = feature;

            featureDiv.appendChild(featureCheckbox);
            featureDiv.appendChild(featureLabel);

            categoryDiv.appendChild(featureDiv);
        });

        listContainer.appendChild(categoryDiv);
    });
}

document.addEventListener('DOMContentLoaded', generateCheckboxes);

function getSelectedFeatures() {
    var selectedFeatures = {};
    var allCategories = document.querySelectorAll('.category-div');

    allCategories.forEach(function(categoryDiv) {
        var categoryName = categoryDiv.querySelector('h5').textContent.trim();
        // Select checkboxes that are not "Select All" and are checked
        var checkboxes = categoryDiv.querySelectorAll('input[type="checkbox"]:not(.select-all-checkbox):checked');
        var selected = Array.from(checkboxes).map(checkbox => checkbox.value);

        if (selected.length > 0) {
            selectedFeatures[categoryName] = selected;
        }
    });

    return selectedFeatures;
}


document.getElementById('submit-button').addEventListener('click', function() {
    var selectedData = getSelectedFeatures();
    globalSelectedData =selectedData;
    // Here you would typically make an AJAX call to send the data to your server
    // For example: sendSelectedDataToServer(selectedData);
});

// Select all features
document.getElementById('selectAll').addEventListener('change', function(e) {
    var checkboxes = document.querySelectorAll('.features-list .form-check-input');
    checkboxes.forEach(function(checkbox) {
        checkbox.checked = e.target.checked;
    });
});

function toggleSelectAll(checked) {
    var checkboxes = document.querySelectorAll('.features-list .form-check-input');
    checkboxes.forEach(function(checkbox) {
        checkbox.checked = checked;
    });
}

document.getElementById('submit-button').addEventListener('click', function() {
    // Find all checkbox elements within the features list
    var checkboxes = document.querySelectorAll('.features-list .form-check-input');

    // Initialize an array to hold the names of the selected features
    var selectedFeatures = [];

    // Iterate through the checkboxes to see which ones are checked
    checkboxes.forEach(function(checkbox) {
        if (checkbox.checked) {
            // If a checkbox is checked, add its value to the selected features array
            selectedFeatures.push(checkbox.value);
        }
    });
});



// overlap selection
document.addEventListener('DOMContentLoaded', function() {
    var overlapSlider = document.getElementById("windowOverlap");
    var percent = document.querySelector('.percent');

    overlapSlider.addEventListener('input', function() {
        var value = this.value;
        percent.textContent = value + '%';
    });
});

//This function is working for Window length and rage bar for window length workable
document.addEventListener('DOMContentLoaded', function() {
    var inputField = document.getElementById("window-length");
    var rangeSlider = document.getElementById("window-length-range");
    var errorMessage = document.getElementById("error-message");

    function updateValues(value) {
        if (value > 50) {
            errorMessage.style.display = 'block';
            inputField.classList.add("error");
            value = 50;
        } else {
            errorMessage.style.display = 'none';
            inputField.classList.remove("error");
        }
        inputField.value = value;
        rangeSlider.value = value;
    }

    rangeSlider.oninput = function() {
        updateValues(this.value);
    }

    inputField.oninput = function() {
        updateValues(this.value);
    }
});

//This function is working for Overlap and rage bar for window length workable
document.addEventListener('DOMContentLoaded', function() {
    var inputField = document.getElementById("windowOverlapInput");
    var rangeSlider = document.getElementById("windowOverlap");
    var errorMessage = document.getElementById("errorMessageOverlap");

    function updateValues(value) {
        if (value > 99) {
            errorMessage.style.display = 'block';
            inputField.classList.add("error");
            value = 99;
        } else {
            errorMessage.style.display = 'none';
            inputField.classList.remove("error");
        }
        inputField.value = value;
        rangeSlider.value = value;
    }

    rangeSlider.oninput = function() {
        updateValues(this.value);
    }

    inputField.oninput = function() {
        updateValues(this.value);
    }
});


// Getting value for submit

document.getElementById('submit-button').addEventListener('click', function() {
    // Disable the submit button and enable the cancel button
    this.disabled = true;
    document.getElementById('cancelButton').disabled = false;

    // Reset the progress bar to 0 and show it
    var progressBar = document.getElementById('progressBar');
    //progressBar.style.width = '0%';
    progressBar.setAttribute('aria-valuenow', 0);
    progressBar.classList.add('progress-bar-animated'); // If using Bootstrap, for example

    // Update global data and send it to the server
    updateGlobalData();
    sendDataToServer(progressBar);
});

document.getElementById('cancelButton').addEventListener('click', function() {
    // Enable the submit button and disable the cancel button
    document.getElementById('submit-button').disabled = false;
    this.disabled = true;

    // Optionally handle the cancellation process here
    // For example, aborting an AJAX request, hiding the progress bar, etc.
});

function sendDataToServer(progressBar) {
//    console.log(globalZoomedChannelData);
//    console.log(globalSelectedFeatures);
//    console.log(globalWindowLength);
//    console.log(globalWindowOverlap);
   var csrfToken = getCSRFToken();

    fetch('http://127.0.0.1:8000/ml_apps/api/my_api/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
             'X-CSRFToken': csrfToken
        },
        body: JSON.stringify({
            zoomedChannelData: globalZoomedChannelData,
            selectedFeatures:globalSelectedFeatures,
            windowLength: globalWindowLength,
            windowOverlap: globalWindowOverlap

        })
    })
    .then(response => {
        // Handle response and update progress bar to 100%
        //progressBar.style.width = '100%';
        //progressBar.setAttribute('aria-valuenow', 100);

        //progressBar.classList.remove('progress-bar-animated'); // Stop any animation
        console.log("mta from Ajax success")
        return response.json();
    })
    .then(data => {
        // Handle the data from the response

        updateTabPanel(data,globalZoomedChannelData);
    })
    .catch(error => {
        // Handle any errors
        console.error('Error:', error);
    })
    .finally(() => {
        // Re-enable the submit button, disable the cancel button, and reset/hide the progress bar
        document.getElementById('submit-button').disabled = false;
        document.getElementById('cancelButton').disabled = true;
        //progressBar.style.width = '0%';
       // progressBar.setAttribute('aria-valuenow', 0);
        // Hide the progress bar if needed
    });
}

function updateTabPanel(data, globalZoomedChannelData) {
    var tabContainer = document.getElementById('innerTab');
    var contentContainer = document.getElementById('innerTabContent');

    // Generate unique IDs
    var timestamp = Date.now();
    var tabId = 'resultTab' + timestamp;
    var contentId = 'resultContent' + timestamp;
    var plotDivIdSSM = 'plotDivSSM' + timestamp;
    var plotDivIdNOV = 'plotDivNOV' + timestamp;
    var tempZoomedChannelData = 'tempZoomedChannelData' + timestamp;

    // Map globalZoomedChannelData to plotlyTraces
    var plotlyTraces = globalZoomedChannelData.map(function(channelData) {
        var xValues = channelData.data.map((_, index) => index);
        return {
            x: xValues,
            y: channelData.data,
            type: 'scatter',
            mode: 'lines+markers',
            name: channelData.name
        };
    });

    // Create a new tab
    var newTab = document.createElement('li');
    newTab.className = 'nav-item';
    newTab.innerHTML = `<a class="nav-link" id="${tabId}" data-toggle="tab" href="#${contentId}" role="tab">Result ${timestamp}</a>`;
    tabContainer.appendChild(newTab);

    // Create new content pane for the tab
    var newContent = document.createElement('div');
    newContent.className = 'tab-pane fade';
    newContent.id = contentId;
    newContent.role = 'tabpanel';
    newContent.innerHTML = `
        <div class="row">
            <div class="col-md-9  text-center">
                <div id="${tempZoomedChannelData}"></div>
                <div id="${plotDivIdSSM}"></div>
            </div>
        </div>
         <div class="row">
            <div class="col-md-9  text-center">
                <div id="${plotDivIdNOV}"></div>
            </div>
            <div class="col-md-3  text-center">
                <input type="range" class="vertical-slider" id="scroller${timestamp}" min="0" max="10" step="0.1" value="5" oninput="updateHorizontalLine(this.value, '${plotDivIdNOV}', '${timestamp}')" orient="vertical" style="height: 200px;" />
                <button onclick="printNearestPoint('${plotDivIdNOV}', '${JSON.stringify(data.nov_ssm)}')">Print Nearest Point</button>
            </div>
        </div>
    `;
    contentContainer.appendChild(newContent);

    // Extract data for plotting
    var ssmData = data.ssm_data;
    var novData = data.nov_ssm;

    // Trace for SSM data
    var traceSSM = {
        z: ssmData,
        type: 'heatmap',
        name: 'SSM'
    };

    // Trace for NOV SSM data
    var traceNOV = {
        x: Array.from(novData.keys()),
        y: novData,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'NOV SSM'
    };

    var layoutSSM = {
        title: 'SSM Plot'
    };

    var layoutNOV = {
        title: 'NOV SSM Plot',
        xaxis: { title: 'Index' },
        yaxis: { title: 'Value', range: [Math.min(...novData) - 1, Math.max(...novData) + 1] },
        shapes: [{
            type: 'line',
            x0: 0,
            y0: 1, // Initial y-value of the horizontal line
            x1: Math.max(...novData.keys()),
            y1: 1, // The same y0 value to keep the line horizontal
            line: { color: 'red', width: 3 }
        }]
    };

    // Render the plots
    Plotly.newPlot(tempZoomedChannelData, plotlyTraces, {title: 'Global Zoomed Data'}); // Added title for clarity
    Plotly.newPlot(plotDivIdSSM, [traceSSM], layoutSSM);
    Plotly.newPlot(plotDivIdNOV, [traceNOV], layoutNOV);

    // Make the new tab active
    $('#self-similarity-matrix').tab('show');
    $(`#${tabId}`).tab('show');
}

function updateHorizontalLine(value, plotDivId, timestamp) {
    var scrollerId = `scroller${timestamp}`;
    var scroller = document.getElementById(scrollerId);
    var update = {
        'shapes[0].y0': parseFloat(scroller.value),
        'shapes[0].y1': parseFloat(scroller.value)
    };
    Plotly.relayout(plotDivId, update);
}

function printNearestPoint(plotDivId, novDataString) {
    // Assume novData is an array of numbers and parse it from the provided string
    var novData = JSON.parse(novDataString.replace(/&quot;/g, "\""));
    var scroller = document.getElementById(`scroller${plotDivId.replace('plotDivNOV', '')}`);
    var yValue = parseFloat(scroller.value);
    var closestIndex = -1;
    var closestDistance = Infinity;

    // Loop through the novData to find the closest point
    novData.forEach((value, index) => {
        var distance = Math.abs(value - yValue);
        if (distance < closestDistance) {
            closestDistance = distance;
            closestIndex = index;
        }
    });

    if (closestIndex !== -1) {
        // Found the nearest point
        var nearestPoint = { x: closestIndex, y: novData[closestIndex] };
        console.log(`Nearest point to y=${yValue}:`, nearestPoint);

        // Download the nearest point as a JSON file
        var dataStr = `data:text/json;charset=utf-8,${encodeURIComponent(JSON.stringify(nearestPoint))}`;
        var downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", "nearest_point.json");
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
    } else {
        console.log("No data points found.");
    }
}


function updateGlobalData() {
    globalSelectedFeatures = getSelectedFeatures(); // Assuming this function gets the checkboxes' data
    globalWindowLength = document.getElementById('window-length').value;
    globalWindowOverlap = document.getElementById('windowOverlap').value; // Make sure the ID is correct here
}

// getting csrf token
function getCSRFToken() {
    let csrfToken = '';
    const cookies = document.cookie.split(';');
    for (let cookie of cookies) {
        const [key, value] = cookie.trim().split('=');
        if (key === 'csrftoken') {
            csrfToken = value;
            break;
        }
    }
    return csrfToken;
}
