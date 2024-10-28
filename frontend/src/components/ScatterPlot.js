import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  ScatterController
} from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';

ChartJS.register(
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  ScatterController,
  zoomPlugin
);

// Define getPointColor outside the component
const getPointColor = (continent) => {
  const colorMap = {
    'North_America': 'rgba(255, 99, 132, 0.6)',
    'South_America': 'rgba(54, 162, 235, 0.6)',
    'Europe': 'rgba(255, 206, 86, 0.6)',
    'Africa': 'rgba(75, 192, 192, 0.6)',
    'Asia': 'rgba(153, 102, 255, 0.6)',
    'Oceania': 'rgba(255, 159, 64, 0.6)'
  };
  return colorMap[continent] || 'rgba(128, 128, 128, 0.6)';
};

const ScatterPlot = React.memo(({ 
  onPointSelect, 
  selectedPoint, 
  selectedCountry, 
  selectedCity,
  onZoomChange,
  currentZoom,
  selectedPCX = 1,
  selectedPCY = 2
}) => {
  const [plotData, setPlotData] = useState([]);
  const [isFlashing, setIsFlashing] = useState(false);
  const [plotDataRange, setPlotDataRange] = useState({
    xMin: 0,
    xMax: 0,
    yMin: 0,
    yMax: 0
  });
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  const flashIntervalRef = useRef(null);
  const containerRef = useRef(null);

  const continentColors = useMemo(() => ({
    'NORTH_AMERICA': 'rgba(255, 99, 132, 0.4)',    // Red
    'SOUTH_AMERICA': 'rgba(54, 162, 235, 0.4)',    // Blue
    'EUROPE': 'rgba(75, 192, 192, 0.4)',           // Teal
    'AFRICA': 'rgba(255, 206, 86, 0.4)',           // Yellow
    'ASIA': 'rgba(153, 102, 255, 0.4)',            // Purple
    'AUSTRALIA': 'rgba(255, 159, 64, 0.4)'         // Orange
  }), []);

  const formatCountryName = useCallback((name) => {
    return name.split('_').map(word =>
      word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
    ).join(' ');
  }, []);

  // Custom Legend Component
  const CustomLegend = React.memo(() => (
    <div style={{
      position: 'absolute',
      left: '10px',
      bottom: '10px',
      backgroundColor: 'rgba(255, 255, 255, 0.9)',
      padding: '10px',
      borderRadius: '8px',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
      display: 'flex',
      flexWrap: 'wrap',
      gap: '8px',
      maxWidth: '100%',
      zIndex: 1
    }}>
      {Object.entries(continentColors).map(([continent, color]) => (
        <div key={continent} style={{
          display: 'flex',
          alignItems: 'center',
          fontSize: '12px',
          marginRight: '12px'
        }}>
          <div style={{
            width: '12px',
            height: '12px',
            backgroundColor: color,
            marginRight: '6px',
            borderRadius: '3px'
          }} />
          <span>{continent.replace('_', ' ')}</span>
        </div>
      ))}
    </div>
  ));

  // Fetch data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:8000/pca_data');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();

        // Transform the data to include x and y coordinates
        const transformedData = result.data.map(point => ({
          ...point,
          x: point.pcs[`PC${selectedPCX}`],
          y: point.pcs[`PC${selectedPCY}`]
        }));

        const xValues = transformedData.map(d => d.x);
        const yValues = transformedData.map(d => d.y);

        setPlotDataRange({
          xMin: Math.min(...xValues),
          xMax: Math.max(...xValues),
          yMin: Math.min(...yValues),
          yMax: Math.max(...yValues)
        });

        setPlotData(transformedData);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, [selectedPCX, selectedPCY]);

  const chartData = useMemo(() => {
    const highlightedDataset = {
      label: 'Selected Points',
      data: plotData.filter(d => {
        if (selectedPoint) return d.pcs[`PC${selectedPCX}`] === selectedPoint.pcs[`PC${selectedPCX}`] && 
                              d.pcs[`PC${selectedPCY}`] === selectedPoint.pcs[`PC${selectedPCY}`];
        if (selectedCity) return d.city === selectedCity;
        return false;
      }).map(d => ({
        ...d,
        x: d.pcs[`PC${selectedPCX}`],
        y: d.pcs[`PC${selectedPCY}`],
        country: formatCountryName(d.country)
      })),
      backgroundColor: isFlashing ? 'rgba(255, 0, 0, 0.8)' : 'rgba(255, 255, 255, 0.8)',
      borderColor: 'black',
      borderWidth: 2,
      pointRadius: 6,
      pointHoverRadius: 8,
      order: 1
    };

    const baseDataset = {
      label: 'All Points',
      data: plotData.filter(d => {
        if (selectedPoint) return d.pcs[`PC${selectedPCX}`] !== selectedPoint.pcs[`PC${selectedPCX}`] || 
                              d.pcs[`PC${selectedPCY}`] !== selectedPoint.pcs[`PC${selectedPCY}`];
        if (selectedCity) return d.city !== selectedCity;
        return true;
      }).map(d => ({
        ...d,
        x: d.pcs[`PC${selectedPCX}`],
        y: d.pcs[`PC${selectedPCY}`],
        country: formatCountryName(d.country)
      })),
      backgroundColor: plotData.map(d => {
        const continentKey = d.continent === 'OCEANIA' ? 'AUSTRALIA' : d.continent;
        return continentColors[continentKey] || 'rgba(128, 128, 128, 0.5)';
      }),
      borderColor: 'transparent',
      borderWidth: 0,
      pointRadius: 3,
      pointHoverRadius: 5,
      order: 2
    };

    return {
      datasets: [baseDataset, highlightedDataset],
    };
  }, [plotData, selectedPoint, selectedCity, isFlashing, continentColors, formatCountryName, selectedPCX, selectedPCY]);

  const chartOptions = useMemo(() => {
    const padding = 0.1;
    const xRange = plotDataRange.xMax - plotDataRange.xMin;
    const yRange = plotDataRange.yMax - plotDataRange.yMin;
    const xPadding = xRange * padding;
    const yPadding = yRange * padding;

    return {
      scales: {
        x: {
          type: 'linear',
          position: 'bottom',
          min: plotDataRange.xMin - xPadding,
          max: plotDataRange.xMax + xPadding,
          title: {
            display: true,
            text: `Principal Component ${selectedPCX}`,
            font: {
              size: 14
            }
          }
        },
        y: {
          type: 'linear',
          position: 'left',
          min: plotDataRange.yMin - yPadding,
          max: plotDataRange.yMax + yPadding,
          title: {
            display: true,
            text: `Principal Component ${selectedPCY}`,
            font: {
              size: 14
            }
          }
        }
      },
      plugins: {
        tooltip: {
          enabled: true,
          mode: 'nearest',
          intersect: false,
          callbacks: {
            label: (context) => {
              const point = context.raw;
              return [
                `Continent: ${point.continent.replace('_', ' ')}`,
                `Country: ${point.country}`,
                `City: ${point.city || 'N/A'}`
              ];
            },
            title: () => null
          }
        },
        legend: {
          display: false,
        },
        zoom: {
          pan: {
            enabled: true,
            mode: 'xy',
          },
          zoom: {
            wheel: {
              enabled: true,
            },
            pinch: {
              enabled: true
            },
            mode: 'xy',
          },
          limits: {
            x: { min: plotDataRange.xMin - xPadding, max: plotDataRange.xMax + xPadding },
            y: { min: plotDataRange.yMin - yPadding, max: plotDataRange.yMax + yPadding }
          }
        }
      },
      interaction: {
        mode: 'nearest',
        intersect: true,
        axis: 'xy'
      },
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      onClick: (event, elements) => {
        if (elements.length > 0) {
          const index = elements[0].index;
          const dataset = elements[0].datasetIndex;
          const clickedPoint = chartData.datasets[dataset].data[index];
          // Ensure we pass the full point data
          const originalPoint = plotData.find(p => 
            p.longitude === clickedPoint.longitude && 
            p.latitude === clickedPoint.latitude
          );
          onPointSelect(originalPoint || clickedPoint);
        }
      },
    };
  }, [plotDataRange, onPointSelect, chartData.datasets, plotData]);

  // Prevent chart recreation
  useEffect(() => {
    let chart = null;
    
    if (plotData.length > 0 && canvasRef.current) {
      if (!chartRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        chart = new ChartJS(ctx, {
          type: 'scatter',
          data: chartData,
          options: chartOptions
        });
        chartRef.current = chart;
      } else {
        chart = chartRef.current;
        chart.data = chartData;
        chart.options = chartOptions;
        chart.update('none');
      }
    }

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, [plotData, chartData, chartOptions]);

  // Separate effect for updating data
  useEffect(() => {
    if (chartRef.current) {
      chartRef.current.data = chartData;
      chartRef.current.update('none');
    }
  }, [chartData]);

  // Always center on selected point while maintaining zoom
  useEffect(() => {
    if (selectedPoint && chartRef.current) {
      const chart = chartRef.current;
      const xScale = chart.scales.x;
      const yScale = chart.scales.y;
      
      // Get current zoom level (range)
      const currentXRange = xScale.max - xScale.min;
      const currentYRange = yScale.max - yScale.min;
      
      // Calculate new boundaries to center the point
      const newXMin = selectedPoint.x - (currentXRange / 2);
      const newXMax = selectedPoint.x + (currentXRange / 2);
      const newYMin = selectedPoint.y - (currentYRange / 2);
      const newYMax = selectedPoint.y + (currentYRange / 2);
      
      // Set new scale boundaries
      chart.scales.x.min = newXMin;
      chart.scales.x.max = newXMax;
      chart.scales.y.min = newYMin;
      chart.scales.y.max = newYMax;
      
      // Force update without animation
      chart.update('none');
    }
  }, [selectedPoint]);

  const handleResetZoom = useCallback(() => {
    if (chartRef.current) {
      const chart = chartRef.current;
      const padding = 0.1; // Same padding as in chartOptions
      const xRange = plotDataRange.xMax - plotDataRange.xMin;
      const yRange = plotDataRange.yMax - plotDataRange.yMin;
      const xPadding = xRange * padding;
      const yPadding = yRange * padding;

      // Reset to initial view with padding
      chart.options.scales.x.min = plotDataRange.xMin - xPadding;
      chart.options.scales.x.max = plotDataRange.xMax + xPadding;
      chart.options.scales.y.min = plotDataRange.yMin - yPadding;
      chart.options.scales.y.max = plotDataRange.yMax + yPadding;
      
      chart.update();
    }
  }, [plotDataRange]);

  const handleFlashPoints = useCallback(() => {
    if (flashIntervalRef.current) {
      clearInterval(flashIntervalRef.current);
    }

    let flashCount = 0;
    flashIntervalRef.current = setInterval(() => {
      setIsFlashing(prev => !prev);
      flashCount++;
      if (flashCount >= 6) {
        clearInterval(flashIntervalRef.current);
        setIsFlashing(false);
      }
    }, 200);
  }, []);

  useEffect(() => {
    return () => {
      if (flashIntervalRef.current) {
        clearInterval(flashIntervalRef.current);
      }
    };
  }, []);

  // Use currentZoom when available
  useEffect(() => {
    if (chartRef.current && currentZoom) {
      const chart = chartRef.current;
      chart.scales.x.min = currentZoom.xMin;
      chart.scales.x.max = currentZoom.xMax;
      chart.scales.y.min = currentZoom.yMin;
      chart.scales.y.max = currentZoom.yMax;
      chart.update('none');
    }
  }, [currentZoom]);

  const handleZoomToPoint = useCallback(() => {
    if (selectedPoint && chartRef.current) {
      const chart = chartRef.current;
      
      // Get current ranges
      const xRange = chart.scales.x.max - chart.scales.x.min;
      const yRange = chart.scales.y.max - chart.scales.y.min;
      
      // Update the chart's options directly
      chart.options.scales.x.min = selectedPoint.x - xRange / 2;
      chart.options.scales.x.max = selectedPoint.x + xRange / 2;
      chart.options.scales.y.min = selectedPoint.y - yRange / 2;
      chart.options.scales.y.max = selectedPoint.y + yRange / 2;
      
      // Force a full update
      chart.update();

      console.log('Updated chart position for point:', selectedPoint);
    }
  }, [selectedPoint]);

  const handleClick = (event, elements) => {
    if (elements.length > 0) {
      const pointIndex = elements[0].index;
      const point = plotData[pointIndex];
      onPointSelect(point);
    }
  };

  const baseDataset = {
    data: plotData,
    backgroundColor: (context) => {
      const point = context.raw;
      if (selectedPoint && 
          point.longitude === selectedPoint.longitude && 
          point.latitude === selectedPoint.latitude) {
        return 'rgba(255, 0, 0, 0.8)'; // Bright red for selected point
      }
      if (selectedCountry && point.country === selectedCountry) {
        return selectedCity && point.city === selectedCity 
          ? 'rgba(255, 0, 0, 0.8)' 
          : 'rgba(255, 165, 0, 0.8)'; // Orange for country matches
      }
      return getPointColor(point.continent);
    },
    borderColor: 'transparent',
    borderWidth: 0,
    pointRadius: (context) => {
      const chart = context.chart;
      const xScale = chart.scales.x;
      const yScale = chart.scales.y;
      
      // Calculate zoom level based on the ratio of the original range to current range
      const xZoom = (plotDataRange.xMax - plotDataRange.xMin) / (xScale.max - xScale.min);
      const yZoom = (plotDataRange.yMax - plotDataRange.yMin) / (yScale.max - yScale.min);
      const zoomFactor = Math.max(xZoom, yZoom);
      
      // Increase base radius to 2 (was 1), increases with zoom but caps at 8 (was 6)
      return Math.min(2 * zoomFactor, 8);
    },
    pointHoverRadius: (context) => {
      const chart = context.chart;
      const xScale = chart.scales.x;
      const yScale = chart.scales.y;
      
      const xZoom = (plotDataRange.xMax - plotDataRange.xMin) / (xScale.max - xScale.min);
      const yZoom = (plotDataRange.yMax - plotDataRange.yMin) / (yScale.max - yScale.min);
      const zoomFactor = Math.max(xZoom, yZoom);
      
      // Hover radius is slightly larger than regular radius
      return Math.min(2 * zoomFactor, 8);
    },
    order: 2
  };

  return (
    <div style={{ 
      height: '100%', 
      width: '100%', 
      position: 'relative',
      boxSizing: 'border-box',
      paddingBottom: '40px'
    }} ref={containerRef}>
      <div style={{
        position: 'absolute',
        top: '10px',
        right: '10px',
        zIndex: 1,
        display: 'flex',
        gap: '10px'
      }}>
        <button
          onClick={handleResetZoom}
          style={{
            padding: '5px 10px',
            backgroundColor: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            fontSize: '12px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
          }}
        >
          Reset Zoom
        </button>
        {(selectedPoint || selectedCity) && (
          <button
            onClick={handleFlashPoints}
            style={{
              padding: '5px 10px',
              backgroundColor: '#4CAF50',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
              fontSize: '12px',
              boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
            }}
          >
            Flash Selected Point(s)
          </button>
        )}
        {selectedPoint && (
          <button
            onClick={handleZoomToPoint}
            style={{
              padding: '5px 10px',
              backgroundColor: '#4CAF50',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
              fontSize: '12px',
              boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
            }}
          >
            Zoom to Point
          </button>
        )}
      </div>
      
      <div style={{ 
        position: 'relative', 
        height: '100%', 
        width: '100%',
        boxSizing: 'border-box'
      }}>
        <canvas ref={canvasRef} style={{ display: 'block', height: '100%', width: '100%' }} />
        <CustomLegend />
      </div>
    </div>
  );
});

export default ScatterPlot;
