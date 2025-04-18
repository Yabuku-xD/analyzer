// Main JavaScript file for the dashboard

// Handle filter changes
document.addEventListener('DOMContentLoaded', function() {
    // Get filter elements
    const industryFilter = document.getElementById('industry');
    const timeFilter = document.getElementById('time-period');
    const categoryFilter = document.getElementById('skill-category');
    
    // Add event listeners
    if (industryFilter) {
        industryFilter.addEventListener('change', function() {
            updateCharts();
        });
    }
    
    if (timeFilter) {
        timeFilter.addEventListener('change', function() {
            updateCharts();
        });
    }
    
    if (categoryFilter) {
        categoryFilter.addEventListener('change', function() {
            updateCharts();
        });
    }
    
    // Function to update charts based on filters
    function updateCharts() {
        const industry = industryFilter ? industryFilter.value : 'all';
        const timePeriod = timeFilter ? timeFilter.value : 'all';
        const category = categoryFilter ? categoryFilter.value : 'all';
        
        // Update rising skills chart
        if (document.getElementById('rising-skills-chart')) {
            fetch(`/api/rising-skills?industry=${industry}&time=${timePeriod}&category=${category}`)
                .then(response => response.json())
                .then(data => {
                    Plotly.react('rising-skills-chart', [{
                        x: data.skills,
                        y: data.values,
                        type: 'bar',
                        marker: {
                            color: '#0055ff'
                        }
                    }], {
                        margin: { t: 10, r: 10, b: 50, l: 50 },
                        yaxis: {
                            title: 'Growth Rate',
                            gridcolor: 'rgba(255,255,255,0.1)'
                        },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: {
                            color: 'rgba(255,255,255,0.8)'
                        }
                    });
                });
        }
        
        // Update declining skills chart
        if (document.getElementById('declining-skills-chart')) {
            fetch(`/api/declining-skills?industry=${industry}&time=${timePeriod}&category=${category}`)
                .then(response => response.json())
                .then(data => {
                    Plotly.react('declining-skills-chart', [{
                        x: data.skills,
                        y: data.values,
                        type: 'bar',
                        marker: {
                            color: '#ff5555'
                        }
                    }], {
                        margin: { t: 10, r: 10, b: 50, l: 50 },
                        yaxis: {
                            title: 'Decline Rate',
                            gridcolor: 'rgba(255,255,255,0.1)'
                        },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: {
                            color: 'rgba(255,255,255,0.8)'
                        }
                    });
                });
        }
        
        // Update skill dropdown
        if (document.getElementById('skill-detail-dropdown')) {
            fetch(`/api/skills?category=${category}`)
                .then(response => response.json())
                .then(data => {
                    const dropdown = document.getElementById('skill-detail-dropdown');
                    
                    // Clear existing options
                    dropdown.innerHTML = '<option value="">Select a skill...</option>';
                    
                    // Add new options
                    data.skills.forEach(skill => {
                        const option = document.createElement('option');
                        option.value = skill;
                        option.textContent = skill;
                        dropdown.appendChild(option);
                    });
                });
        }
    }
});

// Add subtle animation effects
document.addEventListener('DOMContentLoaded', function() {
    // Animate chart containers on hover
    const chartContainers = document.querySelectorAll('.chart-container');
    chartContainers.forEach(container => {
        container.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.boxShadow = '0 8px 30px rgba(0, 0, 0, 0.3)';
        });
        
        container.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = 'none';
        });
    });
});