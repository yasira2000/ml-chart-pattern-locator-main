'use client';

import React from 'react';

interface TimelineRangeSelectorProps {
    visibleStartDate: Date;
    visibleEndDate: Date;
    onRangeSelect: (startDate: Date, endDate: Date) => void;
    isSelecting: boolean;
    title: string;
    buttonText: string;
}

const TimelineRangeSelector: React.FC<TimelineRangeSelectorProps> = ({
    visibleStartDate,
    visibleEndDate,
    onRangeSelect,
    isSelecting,
    title,
    buttonText
}) => {
    const [selectedStartDate, setSelectedStartDate] = React.useState<Date | null>(null);
    const [selectedEndDate, setSelectedEndDate] = React.useState<Date | null>(null);

    const formatDate = (date: Date | null): string => {
        if (!date || isNaN(date.getTime())) return '';
        return date.toISOString().split('T')[0];
    };

    const handleStartDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const date = new Date(e.target.value);
        if (!isNaN(date.getTime())) {
            setSelectedStartDate(date);
            if (selectedEndDate && date > selectedEndDate) {
                setSelectedEndDate(date);
            }
        }
    };

    const handleEndDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const date = new Date(e.target.value);
        if (!isNaN(date.getTime())) {
            setSelectedEndDate(date);
            if (selectedStartDate && date < selectedStartDate) {
                setSelectedStartDate(date);
            }
        }
    };

    const handleApplyRange = () => {
        if (selectedStartDate && selectedEndDate &&
            !isNaN(selectedStartDate.getTime()) &&
            !isNaN(selectedEndDate.getTime())) {
            onRangeSelect(selectedStartDate, selectedEndDate);
            setSelectedStartDate(null);
            setSelectedEndDate(null);
        }
    };

    // Ensure we have valid dates for the visible range
    const validVisibleStart = !isNaN(visibleStartDate.getTime()) ? visibleStartDate : new Date();
    const validVisibleEnd = !isNaN(visibleEndDate.getTime()) ? visibleEndDate : new Date();

    return (
        <div className="bg-gray-800 p-4 rounded-lg shadow-lg">
            <div className="flex flex-col space-y-4">
                <div className="flex justify-between items-center">
                    <h3 className="text-lg font-medium text-gray-200">{title}</h3>
                    <span className="text-sm text-gray-400">
                        Available Range: {validVisibleStart.toLocaleDateString()} - {validVisibleEnd.toLocaleDateString()}
                    </span>
                </div>

                <div className="flex flex-wrap gap-4">
                    <div className="flex-1 min-w-[200px]">
                        <label className="block text-sm font-medium text-gray-300 mb-1">
                            Start Date
                        </label>
                        <input
                            type="date"
                            value={formatDate(selectedStartDate)}
                            onChange={handleStartDateChange}
                            min={formatDate(validVisibleStart)}
                            max={formatDate(validVisibleEnd)}
                            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                    </div>
                    <div className="flex-1 min-w-[200px]">
                        <label className="block text-sm font-medium text-gray-300 mb-1">
                            End Date
                        </label>
                        <input
                            type="date"
                            value={formatDate(selectedEndDate)}
                            onChange={handleEndDateChange}
                            min={formatDate(selectedStartDate) || formatDate(validVisibleStart)}
                            max={formatDate(validVisibleEnd)}
                            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                    </div>
                    <div className="flex items-end">
                        <button
                            onClick={handleApplyRange}
                            disabled={!selectedStartDate || !selectedEndDate || isSelecting}
                            className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {buttonText}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default TimelineRangeSelector; 