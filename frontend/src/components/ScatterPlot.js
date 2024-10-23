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

const ScatterPlot = React.memo(({ onPointSelect, selectedPoint, selectedCountry, selectedCity }) => {
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


  const continentColors = useMemo(() => ({
    'NORTH_AMERICA': 'rgba(255, 99, 132, 0.5)',    // Red
    'SOUTH_AMERICA': 'rgba(54, 162, 235, 0.5)',    // Blue
    'EUROPE': 'rgba(75, 192, 192, 0.5)',           // Teal
    'AFRICA': 'rgba(255, 206, 86, 0.5)',           // Yellow
    'ASIA': 'rgba(153, 102, 255, 0.5)',            // Purple
    'AUSTRALIA': 'rgba(255, 159, 64, 0.5)'         // Orange (for Australia)
  }), []);

  const formatCountryName = useCallback((name) => {
    return name.split('_').map(word =>
      word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
    ).join(' ');
  }, []);

  // Fetch data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:8000/tsne_data');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();

        // Calculate data ranges
        const xValues = result.data.map(d => d.x);
        const yValues = result.data.map(d => d.y);

        setPlotDataRange({
          xMin: Math.min(...xValues),
          xMax: Math.max(...xValues),
          yMin: Math.min(...yValues),
          yMax: Math.max(...yValues)
        });

        setPlotData(result.data);
        console.log(`Loaded ${result.data.length} data points`);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, []);

  const isPointVisible = useCallback((point, chart) => {
    if (!point || !chart?.scales?.x || !chart?.scales?.y) return false;

    const xScale = chart.scales.x;
    const yScale = chart.scales.y;

    return point.x >= xScale.min &&
      point.x <= xScale.max &&
      point.y >= yScale.min &&
      point.y <= yScale.max;
  }, []);

  const panToPoint = useCallback((point) => {
    if (!chartRef.current || !point) return;

    const chart = chartRef.current;
    if (!isPointVisible(point, chart)) {
      const xScale = chart.scales.x;
      const yScale = chart.scales.y;

      const xRange = xScale.max - xScale.min;
      const yRange = yScale.max - yScale.min;

      const targetX = point.x - xRange / 2;
      const targetY = point.y - yRange / 2;

      const deltaX = xScale.min - targetX;
      const deltaY = yScale.min - targetY;

      const pixelsX = deltaX * xScale.width / xRange;
      const pixelsY = deltaY * yScale.height / yRange;

      chart.pan({ x: pixelsX, y: pixelsY }, undefined, 'default');
      chart.update('none');
    }
  }, [isPointVisible]);


  const chartData = useMemo(() => {
    const highlightedDataset = {
      label: 'Selected Points',
      data: plotData.filter(d => {
        if (selectedPoint) return d.x === selectedPoint.x && d.y === selectedPoint.y;
        if (selectedCity) return d.city === selectedCity;
        return false;
      }).map(d => ({
        ...d,
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
        if (selectedPoint) return d.x !== selectedPoint.x || d.y !== selectedPoint.y;
        if (selectedCity) return d.city !== selectedCity;
        return true;
      }).map(d => ({
        ...d,
        country: formatCountryName(d.country)
      })),
      backgroundColor: plotData.map(d => {
        // Map OCEANIA to AUSTRALIA for color lookup
        const continentKey = d.continent === 'OCEANIA' ? 'AUSTRALIA' : d.continent;
        return continentColors[continentKey] || 'rgba(128, 128, 128, 0.5)';
      }),
      borderColor: 'transparent',
      borderWidth: 0,
      pointRadius: 2,
      pointHoverRadius: 4,
      order: 2
    };

    return {
      datasets: [baseDataset, highlightedDataset],
    };
  }, [plotData, selectedPoint, selectedCity, isFlashing, continentColors, formatCountryName]);

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
        },
        y: {
          type: 'linear',
          position: 'left',
          min: plotDataRange.yMin - yPadding,
          max: plotDataRange.yMax + yPadding,
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
          display: true,
          position: 'top',
          align: 'center',
          labels: {
            padding: 30,
            generateLabels: () => {
              // Create legend entries for all continents except ANTARCTICA
              return Object.entries(continentColors)
                .map(([continent, color]) => ({
                  text: continent.replace('_', ' '),
                  fillStyle: color,
                  strokeStyle: color,
                  lineWidth: 0,
                }));
            }
          }
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
          onPointSelect(clickedPoint);
        }
      },
    };
  }, [plotDataRange, continentColors, chartData.datasets, onPointSelect]);


  // Initialize and update chart
  useEffect(() => {
    if (plotData.length > 0 && canvasRef.current) {
      if (chartRef.current) {
        chartRef.current.destroy();
      }

      const ctx = canvasRef.current.getContext('2d');
      chartRef.current = new ChartJS(ctx, {
        type: 'scatter',
        data: chartData,
        options: chartOptions
      });

      return () => {
        if (chartRef.current) {
          chartRef.current.destroy();
        }
      };
    }
  }, [plotData, chartData, chartOptions]);

  // Handle selected point visibility
  useEffect(() => {
    if (selectedPoint) {
      panToPoint(selectedPoint);
    }
  }, [selectedPoint, panToPoint]);

  const handleResetZoom = useCallback(() => {
    if (chartRef.current) {
      chartRef.current.resetZoom();
    }
  }, []);

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

  return (
    <div style={{ height: '100%', width: '100%', position: 'relative' }}>
      <div style={{
        position: 'absolute',
        top: '10px',
        right: '10px',
        zIndex: 100,
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
      </div>
      <canvas ref={canvasRef} style={{ height: '100%', width: '100%' }} />
    </div>
  );
});

export default ScatterPlot;