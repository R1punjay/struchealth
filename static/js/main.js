// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Update status text color based on health status
    const updateStatusColor = () => {
        const healthStatus = document.getElementById('health-status');
        if (!healthStatus) return;
        
        const status = healthStatus.textContent.toLowerCase();
        
        // Remove all possible status classes
        healthStatus.classList.remove(
            'health-status-excellent', 
            'health-status-good', 
            'health-status-fair', 
            'health-status-poor', 
            'health-status-critical'
        );
        
        // Add the appropriate class
        if (status.includes('excellent')) {
            healthStatus.classList.add('health-status-excellent');
        } else if (status.includes('good')) {
            healthStatus.classList.add('health-status-good');
        } else if (status.includes('fair')) {
            healthStatus.classList.add('health-status-fair');
        } else if (status.includes('poor')) {
            healthStatus.classList.add('health-status-poor');
        } else if (status.includes('critical')) {
            healthStatus.classList.add('health-status-critical');
        }
    };
    
    // Initialize tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    if (typeof bootstrap !== 'undefined') {
        const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
    }
    
    // Handle seismic zone selection to update seismic load slider
    const seismicZoneSelect = document.getElementById('seismic-zone');
    const seismicLoadSlider = document.getElementById('seismic-load');
    
    if (seismicZoneSelect && seismicLoadSlider) {
        seismicZoneSelect.addEventListener('change', function() {
            const zone = this.value;
            let defaultLoad = 0.5; // Default value
            
            // Set default load based on seismic zone
            if (zone === 'Zone I') {
                defaultLoad = 0.1;
            } else if (zone === 'Zone II') {
                defaultLoad = 0.3;
            } else if (zone === 'Zone III') {
                defaultLoad = 0.5;
            } else if (zone === 'Zone IV') {
                defaultLoad = 0.7;
            } else if (zone === 'Zone V') {
                defaultLoad = 0.9;
            }
            
            // Update slider value
            seismicLoadSlider.value = defaultLoad;
            
            // Update displayed value
            const seismicLoadValue = document.getElementById('seismic-load-value');
            if (seismicLoadValue) {
                seismicLoadValue.textContent = defaultLoad;
            }
        });
    }
    
    // Print report functionality
    const addPrintButton = () => {
        const resultsSection = document.getElementById('results-section');
        if (!resultsSection || resultsSection.classList.contains('d-none')) return;
        
        // Check if button already exists
        if (document.getElementById('print-report-btn')) return;
        
        // Create print button
        const printBtn = document.createElement('button');
        printBtn.id = 'print-report-btn';
        printBtn.className = 'btn btn-outline-secondary mt-3 w-100';
        printBtn.innerHTML = '<i class="fas fa-print me-2"></i>Print Report';
        
        // Add click event
        printBtn.addEventListener('click', function() {
            window.print();
        });
        
        // Append to the first card in results section
        const firstCard = resultsSection.querySelector('.card');
        if (firstCard) {
            const cardBody = firstCard.querySelector('.card-body');
            if (cardBody) {
                cardBody.appendChild(printBtn);
            }
        }
    };
    
    // Create observer to watch for changes in the results section
    const resultsObserver = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                const element = mutation.target;
                if (element.id === 'results-section' && !element.classList.contains('d-none')) {
                    // Results section has become visible
                    updateStatusColor();
                    addPrintButton();
                }
            }
        });
    });
    
    // Start observing results section if it exists
    const resultsSection = document.getElementById('results-section');
    if (resultsSection) {
        resultsObserver.observe(resultsSection, { attributes: true });
    }
    
    // Add hover effect for recommendation items
    document.addEventListener('mouseover', function(e) {
        if (e.target && e.target.closest('.list-group-item')) {
            const item = e.target.closest('.list-group-item');
            item.classList.add('bg-light');
        }
    });
    
    document.addEventListener('mouseout', function(e) {
        if (e.target && e.target.closest('.list-group-item')) {
            const item = e.target.closest('.list-group-item');
            item.classList.remove('bg-light');
        }
    });
}); 